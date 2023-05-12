from typing import Any, Tuple

from src.utils.typing import ArrayLike

import matplotlib.pyplot as plt
import torch


def untuple(x: Tuple[Any, Any]) -> Any:
    if isinstance(x, tuple):
        return x[0]
    return x


def visualize_matrix(
    weight: ArrayLike,
    limit_dim: int = 100,
    canvas: plt = plt,
    save_path: str | None = None,
    title: str | None = None,
) -> None:
    weight = torch.stack([w[:limit_dim] for w in weight[:limit_dim]]).cpu()
    limit = max(abs(weight.min().item()), abs(weight.max().item()))
    img = plt.imshow(
        weight, cmap="RdBu", interpolation="nearest", vmin=-limit, vmax=limit
    )
    canvas.colorbar(img, orientation="vertical")
    if title is not None:
        canvas.title(title)
    if save_path is not None:
        canvas.savefig(save_path)
    canvas.show()


def low_rank_approximation(weight: torch.Tensor, rank: int = 10) -> torch.Tensor:
    typecache = weight.dtype
    weight = weight.to(torch.float32)
    svd = weight.svd()
    wgt_est = torch.zeros(weight.shape).to(weight.device)
    for i in range(rank):
        wgt_est += svd.S[i] * (svd.U[:, i][None].T @ svd.V[:, i][None])
    approx_err = torch.dist(weight, wgt_est)
    print(f"rank {rank} >> ", approx_err)
    weight = weight.to(typecache)
    return wgt_est.to(typecache)
