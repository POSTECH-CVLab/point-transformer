import pytorch_lightning as pl
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader

from pointnet2.data import Indoor3DSemSeg, PartNormalDataset
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from pointnet2.utils.timer import Timer

from pointnet2.models.point_transformer_layer import PointTransformerBlock
from pointnet2.models.transition_down import TransitionDown
from pointnet2.models.transition_up import TransitionUp


class Point_Transformer_SemSeg(PointNet2ClassificationSSG):
    def _build_model(self, dim = [6,32,64,128,256,512], output_dim=13, pos_mlp_hidden=64, attn_mlp_hidden=4, k = 16, sampling_ratio = 0.25):
        self.Encoder = nn.ModuleList()
        for i in range(len(dim)-1):
            if i == 0:
                self.Encoder.append(nn.Linear(dim[i], dim[i+1], bias=False))
            else:
                self.Encoder.append(TransitionDown(dim[i], dim[i+1], k, sampling_ratio, fast=True))
            self.Encoder.append(PointTransformerBlock(dim[i+1], k))
        self.Decoder = nn.ModuleList()

        for i in range(5,0,-1):
            if i == 5:
                self.Decoder.append(nn.Linear(dim[i], dim[i], bias=False))
            else:
                self.Decoder.append(TransitionUp(dim[i+1], dim[i]))

            self.Decoder.append(PointTransformerBlock(dim[i], k))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(dim[1], dim[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(dim[1], output_dim, kernel_size=1),
        )

    def forward(self, pointcloud):
        timer = Timer("forward")
        timer.tic()
        xyz, features = self._break_up_pc(pointcloud)
        features = features.transpose(1,2).contiguous()

        l_xyz, l_features = [xyz], [features]

        for i in range(int(len(self.Encoder)/2)):
            if i == 0:
                li_features = self.Encoder[2*i](l_features[i])
                li_xyz = l_xyz[i]
            else:
                li_features, li_xyz = self.Encoder[2*i](l_features[i], l_xyz[i])
            li_features = self.Encoder[2*i+1](li_features, li_xyz)


            l_features.append(li_features)
            l_xyz.append(li_xyz)
            del li_features, li_xyz            
        D_n = int(len(self.Decoder)/2)

        
        for i in range(D_n):
            if i == 0:
                l_features[D_n-i] = self.Decoder[2*i](l_features[D_n-i])
                l_features[D_n-i] = self.Decoder[2*i+1](l_features[D_n-i], l_xyz[D_n-i])
            else:
                l_features[D_n-i], l_xyz[D_n-i] = self.Decoder[2*i](l_features[D_n-i+1], l_xyz[D_n-i+1], l_features[D_n-i], l_xyz[D_n-i])
                l_features[D_n-i] = self.Decoder[2*i+1](l_features[D_n-i], l_xyz[D_n-i])
                
        del l_features[0], l_features[1:], l_xyz
        out = self.fc_layer(l_features[0].transpose(1,2).contiguous())
        timer.toc()
        return out

    def prepare_data(self):
        self.train_dset = Indoor3DSemSeg(self.hparams["num_points"], train=True, download=False)
        self.val_dset = Indoor3DSemSeg(self.hparams["num_points"], train=False, download=False)


class Point_Transformer_PartSeg(Point_Transformer_SemSeg):
    def _build_model(self, dim = [3,32,64,128,256,512], output_dim=50, pos_mlp_hidden=64, attn_mlp_hidden=4, k = 16, sampling_ratio = 0.25):
        self.Encoder = nn.ModuleList()
        for i in range(len(dim)-1):
            if i == 0:
                self.Encoder.append(nn.Linear(dim[i], dim[i+1], bias=False))
            else:
                self.Encoder.append(TransitionDown(dim[i], dim[i+1], k, sampling_ratio, fast=True))
            self.Encoder.append(PointTransformerBlock(dim[i+1], k))
        self.Decoder = nn.ModuleList()

        for i in range(5,0,-1):
            if i == 5:
                self.Decoder.append(nn.Linear(dim[i], dim[i], bias=False))
            else:
                self.Decoder.append(TransitionUp(dim[i+1], dim[i]))

            self.Decoder.append(PointTransformerBlock(dim[i], k))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(dim[1], dim[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(dim[1], output_dim, kernel_size=1),
        )

    def prepare_data(self):
        self.train_dset = PartNormalDataset(self.hparams["num_points"], 'train')
        self.val_dset = PartNormalDataset(self.hparams["num_points"], 'val')

