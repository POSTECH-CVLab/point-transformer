from torch import nn, einsum
#import pytorch_lightning as pl
#from einops import repeat

# classes

class PointTransformerBlock(nn.Module):

    def __init__(self, dim, pos_mlp_hidden = 64, attn_mlp_hidden = 4):
        super().__init__()
        self.prev_linear = nn.Linear(dim, dim)
        self.attn = PointTransformerLayer(dim = dim, pos_mlp_hidden = pos_mlp_hidden, attn_mlp_hidden = attn_mlp_hidden)
        self.final_linear = nn.Linear(dim, dim)
        
    def forward(self, x, pos):
        
        x = self.prev_linear(x) 
        x_out = self.attn(x, pos)
        x_out = self.final_linear(x_out)
        x_out += x
        return x_out, pos
        
        
class PointTransformerLayer(nn.Module):
    def __init__(self, dim, pos_mlp_hidden = 64, attn_mlp_hidden = 4):
        super().__init__()
        #self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        
        
        # position encoding 
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden, dim),
        )

    def forward(self, x, pos):
        #n = x.shape[1]

        # queries, keys, values
        #q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        #print(q.size(), k.size(), v.size())
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # relative positional embeddings(position encoding)
        rel_pos = pos[:, :, None] - pos[:, None, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None] - k[:, None, :]

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb).squeeze(dim = -1)

        # expand transformed features and add relative positional embeddings
        # ev = repeat(v, 'b j d -> b i j d', i = n)
        v = v + rel_pos_emb

        # attention
        attn = sim.softmax(dim = -2)
        
        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
                
        return agg