from dataclasses import dataclass

from src import functional

import torch


@dataclass(frozen=True, kw_only=True)
class EditResult:
    """Edited LM output."""

    predicted_tokens: list[functional.PredictedToken]
    model_logits: torch.Tensor
    model_generations: list[str]


@dataclass(frozen=True, kw_only=True)
class LinearRelationEditResult(EditResult):
    """Outputs of a linear relation editor."""
