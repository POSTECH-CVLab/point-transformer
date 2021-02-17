import torch.nn as nn

from point_transformer_layer import PointTransformerBlock, PointTransformerLayer
from transition_down import TransitionDown
from transition_up import TransitionUp

class PointTransformerPartSeg(nn.Module):
    def __init__(self, N):
        super().__init__()
        
        self.MLP0 = nn.Linear(3, 32)
        self.PTB0 = PointTransformerBlock(dim=32)
        
        # Encoder - Stage 1
        self.TD1 = TransitionDown(in_channels=32,
                                  out_channels=64,
                                  k=16,
                                  sampling_ratio=0.25)
        self.PTB1_encode = PointTransformerBlock(dim=64)
        
        # Encoder - Stage 2
        self.TD2 = TransitionDown(in_channels=64,
                                  out_channels=128,
                                  k=16,
                                  sampling_ratio=0.25)
        self.PTB2_encode = PointTransformerBlock(dim=128)
        
        # Encoder - Stage 3
        self.TD3 = TransitionDown(in_channels=128,
                                  out_channels=256,
                                  k=16,
                                  sampling_ratio=0.25)
        self.PTB3_encode = PointTransformerBlock(dim=256)
        
        # Encoder - Stage 4
        self.TD4 = TransitionDown(in_channels=256,
                                  out_channels=512,
                                  k=16,
                                  sampling_ratio=0.25)
        self.PTB4_encode = PointTransformerBlock(dim=512)
        
        self.MLP1 = nn.Linear(512, 512)
        self.PTB_mid = PointTransformerBlock(dim=512)
        
        # Decoder - Stage 1
        self.TU1 = TransitionUp(in_channels=512,
                                out_channels=256)
        self.PTB1_decode = PointTransformerBlock(dim=256)
        
        # Decoder - Stage 2
        self.TU2 = TransitionUp(in_channels=256,
                                out_channels=128)
        self.PTB2_decode = PointTransformerBlock(dim=128)
        
        # Decoder - Stage 3
        self.TU3 = TransitionUp(in_channels=128,
                                out_channels=64)
        self.PTB3_decode = PointTransformerBlock(dim=64)
        
        # Decoder - Stage 4
        self.TU4 = TransitionUp(in_channels=64,
                                out_channels=32)
        self.PTB4_decode = PointTransformerBlock(dim=32)
        
        self.MLP2 = nn.Sequential()
        
    
    def forward(self, points):
        '''
        inputs
            points: (B, N, 3) shaped torch Tensor
            
        outputs
            
        '''
        N = points.shape[1]
        
        # Stage 0
        p = points                  # p: (B, N, 3) shaped torch Tensor
        x = MLP0(points)            # x: (B, N, 32) shaped torch Tensor
        x1, p1 = PTB0(x, p)         # x1: (B, N, 32) shaped torch Tensor
                                    # p1: (B, N, 3) shaped torch Tensor
            
        # Encoder - Stage 1
        x, p = TD1(x1, p1)          # x: (B, N/4, 64) shaped torch Tensor
                                    # p: (B, N/4, 3) shaped torch Tensor
        x2, p2 = PTB1_encode(x, p)  # x2: (B, N/4, 64) shaped torch Tensor
                                    # p2: (B, N/4, 3) shaped torch Tensor
        
        # Encoder - Stage 2
        x, p = TD2(x2, p2)          # x: (B, N/16, 128) shaped torch Tensor
                                    # p: (B, N/16, 3) shaped torch Tensor
        x3, p3 = PTB2_encode(x, p)  # x3: (B, N/16, 128) shaped torch Tensor
                                    # p3: (B, N/16, 3) shaped torch Tensor
            
        # Encoder - Stage 3
        x, p = TD3(x3, p3)          # x: (B, N/64, 256) shaped torch Tensor
                                    # p: (B, N/64, 3) shaped torch Tensor
        x4, p4 = PTB3_encode(x, p)  # x4: (B, N/64, 256) shaped torch Tensor
                                    # p4: (B, N/64, 3) shaped torch Tensor
            
        # Encoder - Stage 4
        x, p = TD4(x4, p4)
        x, p = PTB4_encode(x, p)    # x: (B, N/256, 512) shaped torch Tensor
                                    # p: (B, N/256, 3) shaped torch Tensor 
            
        # Middle stage
        x = MLP1(x)
        x, p = PTB_mid(x, p)        # x: (B, N/256, 512) shaped torch Tensor
                                    # p: (B, N/256, 3) shaped torch Tensor
        
        # Decoder - Stage 1
        x, p = TU1(x, p, x4, p4)
        x, p = PTB1_decode(x, p)    # x: (B, N/64, 256) shaped torch Tensor
                                    # p: (B, N/64, 3) shaped torch Tensor
        
        # Decoder - Stage 2
        x, p = TU2(x, p, x3, p3)    
        x, p = PTB2_decode(x, p)    # x: (B, N/16, 128) shaped torch Tensor
                                    # p: (B, N/16, 3) shaped torch Tensor
        
        # Decoder - Stage 3
        x, p = TU3(x, p, x2, p2)
        x, p = PTB3_decode(x, p)    # x: (B, N/4, 64) shaped torch Tensor
                                    # p: (B, N/4, 3) shaped torch Tensor
        
        # Decoder - Stage 4
        x, p = TU4(x, p, x1, p1)
        x, p = PTB4_decode(x, p)    # x: (B, N, 32) shaped torch Tensor (pointwise feature vectors)
                                    # p: (B, N, 3) shaped torch Tensor (points)
        
        # Output MLP Stage
        