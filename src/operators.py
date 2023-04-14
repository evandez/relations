from dataclasses import dataclass
from typing import Any

from src import data, functional, models
from src.utils import tokenizer_utils

import baukit
import torch


@dataclass(frozen=True, kw_only=True)
class PredictedObject:
    token: str
    prob: float


@dataclass(frozen=True, kw_only=True)
class RelationOutput:
    predictions: list[PredictedObject]


@dataclass(frozen=True, kw_only=True)
class LinearRelationOutput(RelationOutput):
    h: torch.Tensor
    z: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class RelationOperator:
    mt: models.ModelAndTokenizer

    def __call__(self, subject: str, **kwargs: Any) -> RelationOutput:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class LinearRelationOperator(RelationOperator):
    weight: torch.Tensor
    bias: torch.Tensor
    h_layer: int
    z_layer: int
    prompt_template: str

    def __call__(
        self,
        subject: str,
        prompt_template: str | None = None,
        subject_token_index: int = -1,
        return_top_k: int = 5,
        h: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LinearRelationOutput:
        if kwargs:
            raise ValueError(f"unexpected kwargs: {kwargs}")
        if prompt_template is None:
            prompt_template = self.prompt_template

        if h is None:
            prompt = prompt_template.format(subject)
            inputs = self.mt.tokenizer(
                prompt, return_tensors="pt", return_offsets_mapping=True
            ).to(self.mt.model.device)
            offset_mapping = inputs.pop("offset_mapping")
            subject_i, subject_j = tokenizer_utils.find_token_range(
                prompt, subject, offset_mapping=offset_mapping[0]
            )
            h_index = tokenizer_utils.determine_token_index(
                subject_i, subject_j, subject_token_index
            )

            h_layer_name = f"transformer.h.{self.h_layer}"
            with baukit.Trace(self.mt.model, h_layer_name) as ret:
                self.mt.model(**inputs)
            h = ret.output[0][:, h_index]

        z = h.mm(self.weight.t()) + self.bias
        logits = self.mt.lm_head(z)
        dist = torch.softmax(logits.float(), dim=-1)

        topk = dist.topk(dim=-1, k=return_top_k)
        probs = topk.values.view(return_top_k).tolist()
        token_ids = topk.indices.view(return_top_k).tolist()
        words = [self.mt.tokenizer.decode(token_id) for token_id in token_ids]

        return LinearRelationOutput(
            predictions=[
                PredictedObject(token=w, prob=p) for w, p in zip(words, probs)
            ],
            h=h,
            z=z,
        )


@dataclass(frozen=True, kw_only=True)
class Estimator:
    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class JacobianEstimator(Estimator):
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

        inputs = self.mt.tokenizer(
            prompt, return_tensors="pt", return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping")

        subject_i, subject_j = tokenizer_utils.find_token_range(
            prompt, subject, offset_mapping=offset_mapping[0]
        )
        h_index = tokenizer_utils.determine_token_index(
            subject_i, subject_j, self.subject_token_index
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
            prompt_template=prompt_template,  # TODO(evan): Should this be a property?
        )
