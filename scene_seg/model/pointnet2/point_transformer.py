import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lib.pointops.functions import pointops


class PointTransformerLayer(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 num_neighbors=16):
        self.out_channels = in_channels if out_channels is None else out_channels
        super(PointTransformerLayer, self).__init__()
        
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
        
    def forward(self, p, x):
        # points, p: (B, N, 3)
        # in_features, x: (B, C_in, N)
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
        y = torch.einsum('b c n k, b c n k -> b c n', n_v, a)
        return y
    
    
# class TransitionDown(nn.Module):
    
#     def __init__(self,
#                  in_channels,
#                  out_channels=None,
#                  stride=4,
#                  num_neighbors=16):
#         self.out_channels = in_channels if out_channels is None else out_channels
#         assert isinstance(stride, int) and stride > 1
#         super(TransitionDown, self).__init__()
        
#         self.grouper = pointops.QueryAndGroup(nsample=num_neighbors)
#         self.mlp = nn.Sequential(
            
#         )
        
    
    
if __name__ == "__main__":
    from time import time

    assert torch.cuda.is_available()    
    B, C, N, K = 2, 3, 1024, 16
    
    p = torch.randn(B, N, 3).cuda()
    x = torch.randn(B, C, N).cuda()

    layer = PointTransformerLayer(C, num_neighbors=K).cuda()
    s = time()
    y = layer(p, x)
    d = time() - s
    print(y.shape)
    print(d)