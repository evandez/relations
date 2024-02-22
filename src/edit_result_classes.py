from dataclasses import dataclass
from typing import Optional

from src import functional

import torch


@dataclass(frozen=True, kw_only=True)
class EditResult:
    """Edited LM output."""

    predicted_tokens: list[functional.PredictedToken]
    model_logits: Optional[torch.Tensor] = None
    model_generations: Optional[list[str]] = None


@dataclass(frozen=True, kw_only=True)
class LinearRelationEditResult(EditResult):
    """Outputs of a linear relation editor."""
