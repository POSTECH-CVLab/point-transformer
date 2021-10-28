from copy import deepcopy

import torch
import torch.nn as nn

from lib.pointops.functions import pointops

EPSILON = 1e-8


class PointTransformerLayer(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 num_neighbors=16):
        super(PointTransformerLayer, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.to_query = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_key = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_value = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_pos_enc = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, self.out_channels, kernel_size=1)
        )
        self.to_attn = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        )
        
        self.key_grouper = pointops.QueryAndGroup(nsample=num_neighbors, return_idx=True)
        self.value_grouper = pointops.QueryAndGroup(nsample=num_neighbors, use_xyz=False)
        self.softmax = nn.Softmax(dim=-1) # (B, C_out, N, K)
        
    def forward(self, px):
        # points, p: (B, N, 3)
        # in_features, x: (B, C_in, N)
        p, x = px
        
        # query, key, and value
        q = self.to_query(x) # (B, C_out, N)
        k = self.to_key(x) # (B, C_out, N)
        v = self.to_value(x) # (B, C_out, N)
        
        # neighbor search
        n_k, _, n_idx = self.key_grouper(xyz=p, features=k) # (B, 3+C_out, N, K)
        n_v, _ = self.value_grouper(xyz=p, features=v, idx=n_idx.int()) # (B, C_out, N, K)
        
        # relative positional encoding
        n_r = self.to_pos_enc(n_k[:, 0:3, :, :]) # (B, C_out, N, K)
        n_v = n_v + n_r
        
        # self-attention
        a = self.to_attn(q.unsqueeze(-1) - n_k[:, 3:, :, :] + n_r) # (B, C_out, N, K)
        a = self.softmax(a)
        y = torch.sum(n_v * a, dim=-1, keepdim=False)
        return [p, y]
    
    
class PointTransformerBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 num_neighbors=16):
        super(PointTransformerBlock, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.linear1 = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.transformer = PointTransformerLayer(self.out_channels, num_neighbors=num_neighbors)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.linear2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, px):
        p, x = px
    
        y = self.relu(self.bn1(self.linear1(x)))
        y = self.relu(self.bn(self.transformer([p, y])[1]))
        y = self.bn2(self.linear2(y))
        y += x
        y = self.relu(y)
        return [p, y]


class TransitionDown(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 stride=4,
                 num_neighbors=16):
        assert stride > 1
        super(TransitionDown, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.stride = stride
        self.grouper = pointops.QueryAndGroup(nsample=num_neighbors, use_xyz=True)
        self.mlp = nn.Sequential(
            nn.Conv2d(3 + in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d((1, num_neighbors))
        
    def forward(self, p1x):
        # points, p: (B, N, 3)
        # in_features, x: (B, C_in, N)
        p1, x = p1x
        
        # furthest point sampling and neighbor search
        M = x.shape[-1] // self.stride
        p1_trans = p1.transpose(1, 2).contiguous() # (B, 3, N)
        p2 = pointops.gathering(p1_trans, pointops.furthestsampling(p1, M)).transpose(1, 2).contiguous()
        n_x, _ = self.grouper(xyz=p1, new_xyz=p2, features=x) # (B, 3 + C_in, M, K)
        
        # mlp and local max pooling
        n_y = self.mlp(n_x) # (B, C_out, M, K)
        y = self.max_pool(n_y).squeeze(-1) # (B, C_out, M)
        return [p2, y]
    
    
class TransitionUp(nn.Module):
    
    def __init__(self, in_channels, out_channels=None, skip_channels=None):
        super(TransitionUp, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        self.skip_channels = in_channels if skip_channels is None else skip_channels
        
        self.linear1 = nn.Sequential(
            nn.Conv1d(in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Conv1d(self.skip_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, p1x1, p2x2):
        # in_points, p1: (B, N, 3)
        # in_features, x1: (B, C_in, N)
        # skip_points, p2: (B, M, 3)
        # skip_features, x2: (B, C_skip, M)
        p1, x1 = p1x1
        p2, x2 = p2x2
        
        # Three nearest neighbor upsampling
        dist, idx = pointops.nearestneighbor(p2, p1)
        dist_recip = 1.0 / (dist + EPSILON)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        up_x1 = pointops.interpolation(self.linear1(x1), idx, weight)
        
        # aggregation
        y = self.linear2(x2) + up_x1 # (B, C_out, M)
        return [p2, y]


# Just for debugging
class SimplePointTransformerSeg(nn.Module):
    
    def __init__(self, in_channels, num_classes, num_neighbors):
        super(SimplePointTransformerSeg, self).__init__()
        hidden_channels = in_channels * 4
        
        # encoder
        self.in_mlp = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        self.block1 = PointTransformerBlock(hidden_channels, num_neighbors=num_neighbors)
        self.down = TransitionDown(hidden_channels, num_neighbors=num_neighbors)
        self.block2 = PointTransformerBlock(hidden_channels, num_neighbors=num_neighbors)
        
        # decoder
        self.up = TransitionUp(hidden_channels)
        self.block3 = PointTransformerBlock(hidden_channels, num_neighbors=num_neighbors)
        self.out_mlp = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, num_classes, kernel_size=1)
        )
        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else deepcopy(xyz)
        return xyz, features
        
    def forward(self, pc):
        # stride == 1
        p1, x1 = self._break_up_pc(pc)
        x1 = self.in_mlp(x1)
        p1x1 = self.block1([p1, x1])
        
        # stride == 4
        p4x4 = self.down(p1x1)
        p4x4 = self.block2(p4x4)
        
        # stride == 1
        p1y = self.up(p4x4, p1x1)
        p1y = self.block3(p1y)
        y = self.out_mlp(p1y[1])
        return y


if __name__ == "__main__":
    from time import time
    
    assert torch.cuda.is_available()
    B, C_in, C_out, N, K = 4, 6, 20, 1024, 16
    
    model = SimplePointTransformerSeg(C_in, C_out, K).cuda()
    pc = torch.randn(B, N, 3 + C_in).cuda()

    s = time()
    y = model(pc)
    d = time() - s
    
    print("Elapsed time (sec):", d)
    print(y.shape)