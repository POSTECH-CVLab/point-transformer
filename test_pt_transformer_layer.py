import torch
import numpy as np
from pointnet2.models.point_transformer_layer import PointTransformerLayer
from pointnet2.models.point_transformer_layer import PointTransformerBlock

attn = PointTransformerLayer(
    dim = 32,
    pos_mlp_hidden = 64,
    attn_mlp_hidden = 4
)

pt_transformer = PointTransformerBlock(
    dim = 32,
    pos_mlp_hidden = 64,
    attn_mlp_hidden = 4
)

x = torch.randn(1, 16, 32)
pos = torch.randn(1, 16, 3)

x_out = attn.forward(x, pos)

#x_out2 = pt_transformer.forward(x, pos)
x_out2 = pt_transformer(x, pos)

#print(x.shape)
#print(x_out.shape)
print(x_out2[0].shape)
