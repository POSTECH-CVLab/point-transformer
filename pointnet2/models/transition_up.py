import torch.nn as nn

from pointnet2_ops import pointnet2_utils


class TransitionUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True)
        )
        self.lateral_mlp = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x1, p1, x2, p2):
        """
            x1: (B, in_channels, N) torch.Tensor
            p1: (B, N, 3) torch.Tensor
            x2: (B, out_channels, M) torch.Tensor
            p2: (B, M, 3) torch.Tensor
        Note that N is smaller than M because this module upsamples features.
        """
        x1 = self.up_mlp(x1)
        dist, idx= pointnet2_utils.three_nn(p2, p1)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pointnet2_utils.three_interpolate(
            x1, idx, weight
        )
        x2 = self.lateral_mlp(x2)
        return interpolated_feats + x2, p2


if __name__ == '__main__':
    import torch
    
    B, N, M = 4, 1024, 2048
    in_channels = 128
    out_channels = 64

    x1 = torch.randn(B, in_channels, N).cuda()
    p1 = torch.randn(B, N, 3).cuda()
    x2 = torch.randn(B, out_channels, M).cuda()
    p2 = torch.randn(B, M, 3).cuda()

    trans_up = TransitionUp(in_channels, out_channels).cuda()
    y, p3 = trans_up(x1, p1, x2, p2)

    assert torch.all(torch.eq(p3, p2))
    print(y.shape)