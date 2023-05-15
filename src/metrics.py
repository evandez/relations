"""Functions for computing metrics."""
from typing import Sequence

from src import functional
from src.utils.typing import ArrayLike, StrSequence


def recall(predictions: Sequence[StrSequence], targets: StrSequence) -> list[float]:
    """Compute the recall@k for predicted tokens.

    A prediction is considered correct if it is a prefix of the target.
    Insensitive to case and whitespace.

    Args:
        predictions: List of top-k predicted tokens.
        targets: Target tokens. Must be the same length as `predictions`.

    Returns:
        List of [recall@1, recall@2, ..., recall@k].

    """
    _validate_same_length(predictions=predictions, targets=targets)
    if len(predictions) == 0:
        return None

    k = max(map(len, predictions))
    recalls = [0.0] * k
    for topk, target in zip(predictions, targets):
        for i in range(k):
            if functional.any_is_nontrivial_prefix(topk[: i + 1], target):
                recalls[i] += 1

    return [r / len(targets) for r in recalls]


def _validate_same_length(**kwargs: Sequence | ArrayLike) -> None:
    """Validate all batch sizes are the same."""
    lengths = {key: len(seq) for key, seq in kwargs.items()}
    if len(set(lengths.values())) > 1:
        message = f"inconsistent batch sizes:" + "\n\t"
        message += "\n\t".join(f"{key}={length}" for key, length in lengths.items())
        raise ValueError(message)
