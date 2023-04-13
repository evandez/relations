from dataclasses import dataclass
from typing import Any

from src import data, models

import torch


@dataclass(frozen=True, kw_only=True)
class PredictedObject:
    token: str
    prob: float


@dataclass(frozen=True, kw_only=True)
class RelationOutput:
    predictions: list[PredictedObject]


@dataclass(frozen=True, kw_only=True)
class RelationOperator:
    mt: models.ModelAndTokenizer

    def __call__(self, subject: str, **kwargs: Any) -> RelationOutput:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class LinearRelationOutput(RelationOutput):
    h: torch.Tensor
    z: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class LinearRelationOperator(RelationOperator):
    weight: torch.Tensor
    bias: torch.Tensor
    h_layer: int
    z_layer: int

    def __call__(
        self,
        subject: str,
        subject_token_index: int = -1,
        prompt_template: str | None = None,
        h: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LinearRelationOutput:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Estimator:
    def __call__(self, dataset: data.RelationDataset) -> LinearRelationOperator:
        raise NotImplementedError


class JacobianEstimator(Estimator):
    def __call__(self, dataset: data.RelationDataset) -> LinearRelationOperator:
        raise NotImplementedError


class JacobianICLEstimator(Estimator):
    def __call__(self, dataset: data.RelationDataset) -> LinearRelationOperator:
        raise NotImplementedError


class CornerSGDEstimator(Estimator):
    def __call__(self, dataset: data.RelationDataset) -> LinearRelationOperator:
        raise NotImplementedError
