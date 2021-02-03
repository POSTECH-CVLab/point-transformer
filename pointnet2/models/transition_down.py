import torch
import torch.nn as nn
from pointnet2.utils.knn import kNN
from pointnet2_ops import pointnet2_utils


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, k, sampling_ratio):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.sampling_ratio = sampling_ratio
        self.mlp_layer = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.BatchNorm1d(self.out_channels), nn.ReLU(True))

    def forward(self, x, p1):
        '''
        inputs
            x: (n, in_channels) shaped torch Tensor (A set of feature vectors)
            p1: (n, 3) shaped torch Tensor (3D coordinates)
        
        outputs
            y: (n * sampling_ratio, out_channels) shaped torch Tensor
            p2: (n * sampling_ratio, 3) shaped torch Tensor
        '''
        n = p1.shape[0]
        sampled_n = int(n * self.sampling_ratio)

        # 1: Furthest Point Sampling
        p1 = p1.unsqueeze(0)
        sampled_index = pointnet2_utils.furthest_point_sample(p1, sampled_n)
        sampled_index = sampled_index.squeeze(0).long()
        p1 = p1.squeeze(0)
        p2 = p1[sampled_index]

        # 2: kNN & MLP
        neighbors = kNN(p2, p1, self.k)  # neighbors: (sampled_n, k)

        # 2-1: Apply MLP onto each feature
        mlp_x = self.mlp_layer(x)  # mlp_x: (n, out_channels)

        # 2-2: Extract features based on neighbors
        features = mlp_x[neighbors]  # features: (sampled_n, k, out_channels)

        # 3: Local Max Pooling
        y = torch.max(features, dim=1)[0]  # y: (sampled_n, out_channels)

        return y, p2

if __name__ == "__main__":

    N, k, sampling_ratio = 1024, 16, 0.25
    in_channels, out_channels = 64, 128

    x = torch.rand(N, in_channels).cuda()
    p1 = torch.rand(N, 3).cuda()

    print(f"Input: ")
    print(f"x: {x.shape}")
    print(f"p1: {p1.shape}")
    trans_down = TransitionDown(in_channels, out_channels, k, sampling_ratio).cuda()
    
    y, p2 = trans_down(x, p1)

    assert y.shape == (256, 128), f"y should have (256, 128) shape, but has {y.shape}." 
    assert p2.shape == (256, 3), f"p2 should have (256, 3) shape, but has {p2.shape}."
    print(f"Output: ")
    print(f"y: {y.shape}")
    print(f"p2: {p2.shape}")
