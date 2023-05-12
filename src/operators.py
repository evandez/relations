from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Sequence

from src import data, functional, models
from src.utils import tokenizer_utils
from src.utils.typing import ModelInput

import torch


@dataclass(frozen=True, kw_only=True)
class PredictedObject:
    """A predicted object token and its probability under the decoder head."""

    token: str
    prob: float


@dataclass(frozen=True, kw_only=True)
class RelationOutput:
    """Predicted object tokens and their probabilities under the decoder head."""

    predictions: list[PredictedObject]


@dataclass(frozen=True, kw_only=True)
class LinearRelationOutput(RelationOutput):
    """Relation output, the input `h`, and the predicted object hidden state `z`."""

    h: torch.Tensor
    z: torch.Tensor

    def as_relation_output(self) -> RelationOutput:
        return RelationOutput(predictions=self.predictions)


@dataclass(frozen=True, kw_only=True)
class RelationOperator:
    """An abstract relation operator, which maps subjects to objects."""

    def __call__(self, subject: str, **kwargs: Any) -> RelationOutput:
        raise NotImplementedError


SubjectTokenOffsetFn = Callable[[str, str], int]


@dataclass(frozen=True, kw_only=True)
class LinearRelationOperator(RelationOperator):
    """A linear approximation of a relation inside an LM."""

    mt: models.ModelAndTokenizer
    weight: torch.Tensor | None
    bias: torch.Tensor | None
    h_layer: int
    z_layer: int
    prompt_template: str
    subject_token_offset: SubjectTokenOffsetFn | None = None
    metadata: Dict = field(default_factory=dict)

    def __call__(
        self,
        subject: str,
        k: int = 5,
        h: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LinearRelationOutput:
        """Predict the top-k objects for a given subject.

        Args:
            subject: The subject.
            prompt_template: Override for the default prompt template.
            k: Number of objects to return.
            h: Precomputed h, if available.

        Returns:
            Predicted objects and some metadata.

        """
        if kwargs:
            raise ValueError(f"unexpected kwargs: {kwargs}")

        if h is None:
            prompt = self.prompt_template.format(subject)
            offset = _get_offset(self.subject_token_offset, prompt, subject)
            h_index, inputs = _compute_h_index(
                mt=self.mt, prompt=prompt, subject=subject, offset=offset
            )

            [[hs], _] = functional.compute_hidden_states(
                mt=self.mt, layers=[self.h_layer], inputs=inputs
            )
            h = hs[:, h_index]

        z = h
        if self.weight is not None:
            z = z.mm(self.weight.t())
        if self.bias is not None:
            z = z + self.bias

        logits = self.mt.lm_head(z)
        dist = torch.softmax(logits.float(), dim=-1)

        topk = dist.topk(dim=-1, k=k)
        probs = topk.values.view(k).tolist()
        token_ids = topk.indices.view(k).tolist()
        words = [self.mt.tokenizer.decode(token_id) for token_id in token_ids]

        return LinearRelationOutput(
            predictions=[
                PredictedObject(token=w, prob=p) for w, p in zip(words, probs)
            ],
            h=h,
            z=z,
        )


@dataclass(frozen=True, kw_only=True)
class LinearRelationEstimator:
    """Abstract method for estimating a linear relation operator."""

    mt: models.ModelAndTokenizer

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class JacobianEstimator(LinearRelationEstimator):
    """Estimate a linear relation operator as a first-order approximation."""

    h_layer: int
    z_layer: int | None = None
    subject_token_offset: SubjectTokenOffsetFn | None = None

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        # TODO(evandez): Warn if too many samples present?
        prompt_template = relation.prompt_templates[0]
        return self.call_on_sample(relation.samples[0], prompt_template)

    def call_on_sample(
        self, sample: data.RelationSample, prompt_template: str
    ) -> LinearRelationOperator:
        subject = sample.subject

        prompt = prompt_template.format(subject)
        offset = _get_offset(self.subject_token_offset, prompt, subject)
        h_index, inputs = _compute_h_index(
            mt=self.mt, prompt=prompt, subject=subject, offset=offset
        )

        approx = functional.order_1_approx(
            mt=self.mt,
            prompt=prompt,
            h_layer=self.h_layer,
            h_index=h_index,
            z_layer=self.z_layer,
            z_index=-1,
            inputs=inputs,
        )
        return LinearRelationOperator(
            mt=self.mt,
            weight=approx.weight,
            bias=approx.bias,
            h_layer=approx.h_layer,
            z_layer=approx.z_layer,
            prompt_template=prompt_template,
            metadata=approx.misc,
        )


@dataclass(frozen=True, kw_only=True)
class JacobianIclMaxEstimator(LinearRelationEstimator):
    """Jacobian estimator that uses in-context learning."""

    h_layer: int
    z_layer: int | None = None
    subject_token_offset: SubjectTokenOffsetFn | None = None

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        samples = relation.samples

        prompt_template = relation.prompt_templates[0]

        subject_token_offsets = [
            _get_offset(self.subject_token_offset, prompt_template, sample.subject)
            for sample in samples
        ]

        # Estimate the biases, storing the confidence of the target token
        # along the way.
        approxes = []
        confidences = []
        for i, sample in enumerate(samples):
            prompt = prompt_template.format(sample.subject)
            h_index, inputs = _compute_h_index(
                mt=self.mt,
                prompt=prompt,
                subject=sample.subject,
                offset=subject_token_offsets[i],
            )
            approx = functional.order_1_approx(
                mt=self.mt,
                prompt=prompt,
                h_layer=self.h_layer,
                h_index=h_index,
                z_layer=self.z_layer,
                z_index=-1,
                inputs=inputs,
            )
            approxes.append(approx)

            # TODO(evan): Is this space needed? Seems so, but it makes things worse!
            object_token_id = self.mt.tokenizer(" " + sample.object).input_ids[0]
            logits = approx.logits[0, -1]
            logps = torch.log_softmax(logits, dim=-1)
            confidences.append(logps[object_token_id])

        # Now estimate J, using an ICL prompt with the model's most-confident subject
        # used as the training example.
        chosen = torch.stack(confidences).argmax().item()
        assert isinstance(chosen, int)

        sample = samples[chosen]
        subject = sample.subject

        prompt_icl = _make_icl_prompt(
            prompt_template=prompt_template,
            subject=subject,
            examples=samples,
        )
        h_index_icl, inputs_icl = _compute_h_index(
            mt=self.mt,
            prompt=prompt_icl,
            subject=subject,
            offset=subject_token_offsets[chosen],
        )
        approx_icl = functional.order_1_approx(
            mt=self.mt,
            prompt=prompt_icl,
            h_layer=self.h_layer,
            h_index=h_index_icl,
            z_layer=self.z_layer,
            z_index=-1,
            inputs=inputs_icl,
        )

        # Package it all up.
        weight = approx_icl.weight
        bias = torch.stack([approx.bias for approx in approxes]).mean(dim=0)
        operator = LinearRelationOperator(
            mt=self.mt,
            weight=weight,
            bias=bias,
            h_layer=self.h_layer,
            z_layer=approx_icl.z_layer,
            prompt_template=prompt_template,
        )

        return operator


@dataclass(frozen=True)
class JacobianIclMeanEstimator(LinearRelationEstimator):
    h_layer: int
    z_layer: int | None = None
    subject_token_offset: SubjectTokenOffsetFn | None = None
    bias_scale_factor: float | None = 0.5

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        samples = relation.samples

        prompt_template = relation.prompt_templates[0]

        subject_token_offsets = [
            _get_offset(self.subject_token_offset, prompt_template, sample.subject)
            for sample in samples
        ]

        # Estimate the biases, storing the confidence of the target token
        # along the way.
        approxes = []
        for i, sample in enumerate(samples):
            prompt = _make_icl_prompt(
                prompt_template=prompt_template,
                subject=sample.subject,
                examples=samples,
            )

            h_index, inputs = _compute_h_index(
                mt=self.mt,
                prompt=prompt,
                subject=sample.subject,
                offset=subject_token_offsets[i],
            )
            approx = functional.order_1_approx(
                mt=self.mt,
                prompt=prompt,
                h_layer=self.h_layer,
                h_index=h_index,
                z_layer=self.z_layer,
                z_index=-1,
                inputs=inputs,
            )
            approxes.append(approx)

        weight = torch.stack([approx.weight for approx in approxes]).mean(dim=0)
        bias = torch.stack([approx.bias for approx in approxes]).mean(dim=0)

        # TODO(evan): Scaling bias down helps tremendously, but why?
        # Find better way to determine scaling factor.
        if self.bias_scale_factor is not None:
            bias = self.bias_scale_factor * bias

        operator = LinearRelationOperator(
            mt=self.mt,
            weight=weight,
            bias=bias,
            h_layer=self.h_layer,
            z_layer=approxes[0].z_layer,
            prompt_template=prompt_template,
        )

        return operator


@dataclass(frozen=True, kw_only=True)
class CornerGdEstimator(LinearRelationEstimator):
    """Estimate a "corner" of LM's rep space where range is assigned equal prob."""

    h_layer: int

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        result = functional.corner_gd(mt=self.mt, words=list(relation.range))
        return LinearRelationOperator(
            mt=self.mt,
            weight=None,
            bias=result.corner,
            h_layer=self.h_layer,
            z_layer=-1,
            prompt_template="{}",
        )


def _get_offset(
    subject_token_offset: SubjectTokenOffsetFn | None,
    prompt_template: str,
    subject: str,
    default: int = -1,
) -> int:
    """Determine the offset using the (maybe null) offset fn."""
    if subject_token_offset is None:
        return default
    prompt = prompt_template.format(subject)  # No-op if subject is already in prompt.
    return subject_token_offset(prompt, subject)


def _make_icl_prompt(
    *,
    prompt_template: str,
    subject: str,
    examples: Sequence[data.RelationSample],
) -> str:
    others = [x for x in examples if x.subject != subject]
    prompt = (
        "\n".join(prompt_template.format(x.subject) + f" {x.object}." for x in others)
        + "\n"
        + prompt_template.format(subject)
    )
    return prompt


def _compute_h_index(
    *,
    mt: models.ModelAndTokenizer,
    prompt: str,
    subject: str,
    offset: int,
) -> tuple[int, ModelInput]:
    inputs = mt.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(
        mt.model.device
    )
    offset_mapping = inputs.pop("offset_mapping")

    subject_i, subject_j = tokenizer_utils.find_token_range(
        prompt, subject, offset_mapping=offset_mapping[0]
    )
    h_index = tokenizer_utils.offset_to_absolute_index(subject_i, subject_j, offset)

    return h_index, inputs
