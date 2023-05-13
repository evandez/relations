"""Miscellaneous visualization tools."""
from src.utils.typing import ArrayLike, PathLike

import matplotlib.pyplot as plt
import torch


def matrix_heatmap(
    matrix: ArrayLike,
    limit_dim: int = 100,
    canvas: plt = plt,
    save_path: PathLike | None = None,
    title: str | None = None,
) -> None:
    """Plot cross section of matrix as a heatmap."""
    matrix = torch.stack([w[:limit_dim] for w in matrix[:limit_dim]]).cpu()
    limit = max(abs(matrix.min().item()), abs(matrix.max().item()))
    img = plt.imshow(
        matrix, cmap="RdBu", interpolation="nearest", vmin=-limit, vmax=limit
    )
    canvas.colorbar(img, orientation="vertical")
    if title is not None:
        canvas.title(title)
    if save_path is not None:
        canvas.savefig(str(save_path))
    canvas.show()
