from copy import deepcopy

import torch.nn as nn

from model.point_transformer.point_transformer_modules import (
    PointTransformerBlock,
    TransitionDown,
    TransitionUp
)


class PointTransformerSeg(nn.Module):
    
    def __init__(self, c=6, k=16, args=None):
        super(PointTransformerSeg, self).__init__()
        
        self.nsamples = args.get('nsamples', [8, 16, 16, 16, 16])
        self.strides = args.get('strides', [None, 4, 4, 4, 4])
        self.planes = args.get('planes', [32, 64, 128, 256, 512])
        self.blocks = args.get('blocks', [2, 3, 4, 6, 3])
        
        # encoder
        self.in_mlp = nn.Sequential(
            nn.Conv1d(c, self.planes[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.planes[0], self.planes[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True)
        )
        self.enc_layer1 = self._make_layer(self.planes[0], self.blocks[0], nsample=self.nsamples[0])
        self.down1to2 = TransitionDown(self.planes[0], self.planes[1], stride=self.strides[1], num_neighbors=self.nsamples[1])
        self.enc_layer2 = self._make_layer(self.planes[1], self.blocks[1], nsample=self.nsamples[1])
        self.down2to3 = TransitionDown(self.planes[1], self.planes[2], stride=self.strides[2], num_neighbors=self.nsamples[2])
        self.enc_layer3 = self._make_layer(self.planes[2], self.blocks[2], nsample=self.nsamples[2])
        self.down3to4 = TransitionDown(self.planes[2], self.planes[3], stride=self.strides[3], num_neighbors=self.nsamples[3])
        self.enc_layer4 = self._make_layer(self.planes[3], self.blocks[3], nsample=self.nsamples[3])
        self.down4to5 = TransitionDown(self.planes[3], self.planes[4], stride=self.strides[4], num_neighbors=self.nsamples[4])
        self.enc_layer5 = self._make_layer(self.planes[4], self.blocks[4], nsample=self.nsamples[4])
        
        # decoder
        self.dec_mlp = nn.Sequential(
            nn.Conv1d(self.planes[4], self.planes[4], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.planes[4]),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.planes[4], self.planes[4], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.planes[4]),
            nn.ReLU(inplace=True)
        )
        self.dec_layer5 = self._make_layer(self.planes[4], 2, nsample=self.nsamples[4])
        self.up5to4 = TransitionUp(self.planes[4], self.planes[3], self.planes[3])
        self.dec_layer4 = self._make_layer(self.planes[3], 2, nsample=self.nsamples[3])
        self.up4to3 = TransitionUp(self.planes[3], self.planes[2], self.planes[2])
        self.dec_layer3 = self._make_layer(self.planes[2], 2, nsample=self.nsamples[2])
        self.up3to2 = TransitionUp(self.planes[2], self.planes[1], self.planes[1])
        self.dec_layer2 = self._make_layer(self.planes[1], 2, nsample=self.nsamples[1])
        self.up2to1 = TransitionUp(self.planes[1], self.planes[0], self.planes[0])
        self.dec_layer1 = self._make_layer(self.planes[0], 2, nsample=self.nsamples[0])
        self.out_mlp = nn.Sequential(
            nn.Conv1d(self.planes[0], self.planes[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.planes[0], k, kernel_size=1)
        )
        
    def _make_layer(self, planes, blocks, nsample):
        layers = []
        for _ in range(blocks):
            layers.append(PointTransformerBlock(planes, num_neighbors=nsample))
        return nn.Sequential(*layers)
    
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else deepcopy(xyz)
        return xyz, features
    
    def forward(self, pc):
        p1, x1 = self._break_up_pc(pc)
        
        # encoder
        x1 = self.in_mlp(x1)
        p1x1 = self.enc_layer1([p1, x1])
        p2x2 = self.down1to2(p1x1)
        p2x2 = self.enc_layer2(p2x2)
        p3x3 = self.down2to3(p2x2)
        p3x3 = self.enc_layer3(p3x3)
        p4x4 = self.down3to4(p3x3)
        p4x4 = self.enc_layer4(p4x4)
        p5x5 = self.down4to5(p4x4)
        p5, x5 = self.enc_layer5(p5x5)
        
        # decoder
        y = self.dec_mlp(x5)
        p5y = self.dec_layer5([p5, y])
        p4y = self.up5to4(p5y, p4x4)
        p4y = self.dec_layer4(p4y)
        p3y = self.up4to3(p4y, p3x3)
        p3y = self.dec_layer3(p3y)
        p2y = self.up3to2(p3y, p2x2)
        p2y = self.dec_layer2(p2y)
        p1y = self.up2to1(p2y, p1x1)
        p1, y = self.dec_layer1(p1y)
        y = self.out_mlp(y)
        return y
