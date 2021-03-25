import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from point_transformer.data import Indoor3DSemSeg, PartNormalDataset
from point_transformer.models.base import BaseClassification
from point_transformer.utils.timer import Timer

from point_transformer_ops.point_transformer_modules import PointTransformerBlock, TransitionDown, TransitionUp


class PointTransformerSemSegmentation(BaseClassification):
    def _build_model(self, dim=[6,32,64,128,256,512], output_dim=13, pos_mlp_hidden=64, attn_mlp_hidden=4, k=16, sampling_ratio=0.25):
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


class PointTransformerPartSegmentation(PointTransformerSemSegmentation):
    def _build_model(self, dim=[3,32,64,128,256,512], output_dim=50, pos_mlp_hidden=64, attn_mlp_hidden=4, k=16, sampling_ratio=0.25):
      super(PointTransformerPartSegmentation, self)._build_model(dim, output_dim, pos_mlp_hidden, attn_mlp_hidden, k, sampling_ratio)

    def training_step(self, batch, batch_idx):
      pc, cls, labels = batch
      logits = self.forward(pc)
      loss = F.cross_entropy(logits, labels)
      with torch.no_grad():
        ious = (torch.argmax(logits, dim=1) == labels).float().mean(1)
        miou = ious.mean()
      log = dict(train_loss=loss, train_miou=miou)
      return dict(loss=loss, log=log, progress_bar=dict(train_miou=miou))

    def validation_step(self, batch, batch_idx):
      pc, cls, labels = batch
      logits = self.forward(pc)
      loss = F.cross_entropy(logits, labels)
      with torch.no_grad():
        ious = (torch.argmax(logits, dim=1) == labels).float().mean(1)
        miou = ious.mean()
      return dict(val_loss=loss, val_miou=miou)

    def prepare_data(self):
        self.train_dset = PartNormalDataset(self.hparams["num_points"], 'trainval')
        self.val_dset = PartNormalDataset(self.hparams["num_points"], 'test')

    def _build_dataloader(self, dset, mode):
        return DataLoader(
            dset,
            batch_size=self.hparams["batch_size"] if mode == 'train' else 1,
            shuffle=mode == "train",
            num_workers=4,
            pin_memory=True,
            drop_last=mode == "train",
        )