#Reference : https://github.com/qq456cvb/Point-Transformers/blob/master/models/Hengshuang/transformer.py

from torch import nn, einsum
import numpy as np

import torch 
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
#from pointnet2.utils.knn import kNN

# classes


def square_dist(p1, p2):
    return torch.sum((p1[:,:,None]-p2[:,None])**2, dim=-1)

def idx_pt(pts, idx):
    raw_size  = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    #print(idx.size())
    #print(pts.size())
    res = torch.gather(pts, 1, idx[..., None].expand(-1, -1, pts.size(-1)))
    return res.reshape(*raw_size,-1)


        
        
class PointTransformerBlock(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        
        self.prev_linear = nn.Linear(dim, dim)

        self.k = k

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        
        # position encoding 
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim)
        )

        self.final_linear = nn.Linear(dim, dim)

    def forward(self, x, pos):
        # queries, keys, values

        x_pre = x

        dist = square_dist(pos, pos)
        knn_idx = dist.argsort()[:,:,:self.k]
        knn_xyz = idx_pt(pos, knn_idx)

        q = self.to_q(x)
        k = idx_pt(self.to_k(x), knn_idx)
        v = idx_pt(self.to_v(x), knn_idx)
        
        pos_enc = self.pos_mlp(pos[:,:,None]-knn_xyz)

        attn = self.attn_mlp(q[:,:,None]-k+pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)

        agg = einsum('b i j d, b i j d -> b i d', attn, v+pos_enc)

        agg = self.final_linear(agg) + x_pre

        return agg
        
    
if __name__ == "__main__":


    attn = PointTransformerBlock(
        dim = 6, k = 16
    )

    x = torch.randn(24, 1000, 6)
    pos = torch.randn(24, 1000, 3)

    x_out = attn.forward(x, pos)

    print(x_out.shape)

