import logging
from dataclasses import dataclass, field
from typing import Any

from src import data, functional, models
from src.functional import low_rank_approx
from src.utils.typing import Layer

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class RelationOutput:
    """Predicted object tokens and their probabilities under the decoder head."""

    predictions: list[functional.PredictedToken]


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


@dataclass(frozen=True, kw_only=True)
class LinearRelationOperator(RelationOperator):
    """A linear approximation of a relation inside an LM."""

    mt: models.ModelAndTokenizer
    weight: torch.Tensor | None
    bias: torch.Tensor | None
    h_layer: Layer
    z_layer: Layer
    prompt_template: str
    metadata: dict = field(default_factory=dict)

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
            prompt = functional.make_prompt(
                mt=self.mt, prompt_template=self.prompt_template, subject=subject
            )
            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt, prompt=prompt, subject=subject
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
                functional.PredictedToken(token=w, prob=p) for w, p in zip(words, probs)
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

    h_layer: Layer
    z_layer: Layer | None = None

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        for key in ("samples", "prompt_templates"):
            values = getattr(relation, key)
            if len(values) == 0:
                raise ValueError(f"expected at least one value for {key}")
            if len(values) > 1:
                logger.warning(f"relation has > 1 {key}, will use first ({values[0]})")

        subject = relation.samples[0].subject
        prompt_template = relation.prompt_templates[0]
        return self.estimate_for_subject(subject, prompt_template)

    def estimate_for_subject(
        self, subject: str, prompt_template: str
    ) -> LinearRelationOperator:
        prompt = functional.make_prompt(
            mt=self.mt, prompt_template=prompt_template, subject=subject
        )
        h_index, inputs = functional.find_subject_token_index(
            mt=self.mt, prompt=prompt, subject=subject
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
            metadata=approx.metadata,
        )


@dataclass(frozen=True, kw_only=True)
class JacobianIclMaxEstimator(LinearRelationEstimator):
    """Jacobian estimator that uses in-context learning."""

    h_layer: Layer
    z_layer: Layer | None = None

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        samples = relation.samples

        prompt_template = relation.prompt_templates[0]

        # Estimate the biases, storing the confidence of the target token
        # along the way.
        approxes = []
        confidences = []
        for i, sample in enumerate(samples):
            prompt = prompt_template.format(sample.subject)
            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt,
                prompt=prompt,
                subject=sample.subject,
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

        prompt_icl = functional.make_prompt(
            mt=self.mt,
            prompt_template=prompt_template,
            subject=subject,
            examples=samples,
        )
        h_index_icl, inputs_icl = functional.find_subject_token_index(
            mt=self.mt,
            prompt=prompt_icl,
            subject=subject,
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
    h_layer: Layer
    z_layer: Layer | None = None
    bias_scale_factor: float | None = 0.5
    rank: int | None = None  # If None, don't do low rank approximation.

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        samples = relation.samples

        prompt_template = relation.prompt_templates[0]

        # Estimate the biases, storing the confidence of the target token
        # along the way.
        approxes = []
        for sample in samples:
            prompt = functional.make_prompt(
                mt=self.mt,
                prompt_template=prompt_template,
                subject=sample.subject,
                examples=samples,
            )
            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt,
                prompt=prompt,
                subject=sample.subject,
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

        if self.rank is not None:
            weight = low_rank_approx(matrix=weight, rank=self.rank)

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

    h_layer: Layer

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
