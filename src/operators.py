from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True, kw_only=True)
class RelationOperator:
    """An abstract relation operator, which maps subjects to objects."""

    def __call__(self, subject: str, **kwargs: Any) -> RelationOutput:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class LinearRelationOperator(RelationOperator):
    """A linear approximation of a relation inside an LM."""

    mt: models.ModelAndTokenizer
    weight: torch.Tensor
    bias: torch.Tensor
    h_layer: int
    z_layer: int
    prompt_template: str

    def __call__(
        self,
        subject: str,
        prompt_template: str | None = None,
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
        if prompt_template is None:
            prompt_template = self.prompt_template

        if h is None:
            prompt = prompt_template.format(subject)
            h_index, inputs = _compute_h_index(
                mt=self.mt, prompt=prompt, subject=subject
            )

            [[hs], _] = functional.compute_hidden_states(
                mt=self.mt, prompt=prompt, layers=[self.h_layer], inputs=inputs
            )
            h = hs[:, h_index]

        z = h.mm(self.weight.t()) + self.bias
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

    mt: models.ModelAndTokenizer
    h_layer: int
    z_layer: int | None = None
    subject_token_index: int = -1

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        if len(relation.samples) != 1:
            raise ValueError("JacobianEstimator only supports one sample")
        if len(relation.prompt_templates) != 1:
            raise ValueError("JacobianEstimator only supports one prompt template")

        [sample] = relation.samples
        subject = sample.subject
        [prompt_template] = relation.prompt_templates

        prompt = prompt_template.format(subject)
        h_index, inputs = _compute_h_index(mt=self.mt, prompt=prompt, subject=subject)

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
        )


def _compute_h_index(
    *, mt: models.ModelAndTokenizer, prompt: str, subject: str
) -> tuple[int, ModelInput]:
    inputs = mt.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(
        mt.model.device
    )
    offset_mapping = inputs.pop("offset_mapping")

    _, subject_j = tokenizer_utils.find_token_range(
        prompt, subject, offset_mapping=offset_mapping[0]
    )
    h_index = subject_j - 1

    return h_index, inputs
