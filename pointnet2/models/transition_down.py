import torch
import torch.nn as nn
import sys

sys.path.append("../utils")

from knn import kNN, kNN_torch, index_points
from pointnet2_ops import pointnet2_utils


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, k, sampling_ratio, fast=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.sampling_ratio = sampling_ratio
        self.fast = fast
        self.mlp = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, x, p1):
        """
        inputs
            x: (B, N, in_channels) shaped torch Tensor (A set of feature vectors)
            p1: (B, N, 3) shaped torch Tensor (3D coordinates)

        outputs
            y: (B, M, out_channels) shaped torch Tensor
            p2: (B, M, 3) shaped torch Tensor

        M = N * sampling ratio
        """
        B, N, _ = x.shape
        M = int(N * self.sampling_ratio)

        # 1: Furthest Point Sampling
        p1_flipped = p1.transpose(1, 2).contiguous()
        p2 = (
            pointnet2_utils.gather_operation(
                p1_flipped, pointnet2_utils.furthest_point_sample(p1, M)
            )
            .transpose(1, 2)
            .contiguous()
        )  # p2: (B, M, 3)

        # 2: kNN & MLP
        knn_fn = kNN_torch if self.fast else kNN
        neighbors = knn_fn(p2, p1, self.k)  # neighbors: (B, M, k)

        # 2-1: Apply MLP onto each feature
        x_flipped = x.transpose(1, 2).contiguous()
        mlp_x = (
            self.mlp(x_flipped).transpose(1, 2).contiguous()
        )  # mlp_x: (B, N, out_channels)

        # 2-2: Extract features based on neighbors
        features = index_points(mlp_x, neighbors)  # features: (B, M, k, out_channels)

        # 3: Local Max Pooling
        y = torch.max(features, dim=2)[0]  # y: (B, M, out_channels)

        return y, p2


if __name__ == "__main__":
    from timer import Timer
    import numpy as np

    B, N, k, sampling_ratio = 2, 1024, 16, 0.25
    in_channels, out_channels = 64, 128
    M = int(N * sampling_ratio)

    x = torch.rand(B, N, in_channels).cuda()
    p1 = torch.rand(B, N, 3).cuda()

    print(f"Input: ")
    print(f"x: {x.shape}")
    print(f"p1: {p1.shape}")

    t, t_fast = Timer("base"), Timer("fast")
    trans_down = TransitionDown(
        in_channels,
        out_channels,
        k,
        sampling_ratio,
    ).cuda()

    trans_down_fast = TransitionDown(
        in_channels,
        out_channels,
        k,
        sampling_ratio,
        fast=True,
    ).cuda()

    t.tic()
    y, p2 = trans_down(x, p1)
    t.toc()

    t_fast.tic()
    y_fast, p2_fast = trans_down_fast(x, p1)
    t_fast.toc()

    np.testing.assert_allclose(
        p2.detach().cpu().numpy(),
        p2_fast.detach().cpu().numpy(),
    )
    print(f"Output: ")
    print(f"y: {y.shape}, y_fast: {y_fast.shape}")
    print(f"p2: {p2.shape}, p2_fast: {p2_fast.shape}")
