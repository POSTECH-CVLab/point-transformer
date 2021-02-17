from torch import nn
import numpy as np

from point_transformer_layer import PointTransformerBlock
from transition_down import TransitionDown
from transition_up import TransitionUp

class model_encoder(nn.Module):
    
    def __init__(self, dim):
        
        self.init_mlp = nn.Linear(3, dim, bias = False)
        
        self.transformerblock = PointTransformerBlock(dim, pos_mlp_hidden = 64, attn_mlp_hidden = 4)
        
        self.transition_down = TransitionDown()
        
        self.transition_up = TransitionUp()
    
    def forward(self, x):
        
        #1st_layer
        
        
        #2nd_layer
        
        
        #3rd_layer
        
        
        #4th_layer
        
        
        #5th_layer 
        

        
class model_decoder(nn.Module):
    
    def __init__(self, ):
        
        
        
        
    def forward(self, x):