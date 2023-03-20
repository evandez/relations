import torch
import matplotlib.pyplot as plt

def visualize_matrix(weight, limit_dim = 100):
    weight = torch.stack([w[:limit_dim] for w in weight[:limit_dim]]).cpu()
    limit = max(abs(weight.min().item()), abs(weight.max().item()))
    img = plt.imshow(
        weight,
        cmap='RdBu', interpolation='nearest', 
        vmin = -limit, vmax = limit
    )
    plt.colorbar(img, orientation='vertical')
    plt.show()


def low_rank_approximation(weight, rank = 10):
    typecache = weight.dtype
    weight = weight.to(torch.float32)
    svd = weight.svd()
    wgt_est = torch.zeros(weight.shape).to(weight.device)
    for i in range(rank):
        wgt_est += svd.S[i] * (svd.U[:, i][None].T @ svd.V[:, i][None])
    # print(f"approximation error ==> {torch.dist(weight, wgt_est)}")
    approx_err = torch.dist(weight, wgt_est)
    print(f"rank {rank} >> ", approx_err)
    weight = weight.to(typecache)
    return wgt_est.to(typecache)