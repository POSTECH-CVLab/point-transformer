import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from point_transformer.utils.timer import Timer

from point_transformer.models.base import BaseClassification
from point_transformer_ops.point_transformer_modules import PointTransformerBlock, TransitionDown


class PointTransformerClassification(BaseClassification):
    def _build_model(self):
        channels, k, sampling_ratio, num_points = (
            self.hparams["model.channels"],
            self.hparams["model.k"],
            self.hparams["model.sampling_ratio"],
            self.hparams["num_points"],
        )
        channels = list(map(int, channels.split(".")))
        assert len(channels) > 3

        self.prev_block = nn.Sequential(
            nn.Linear(3, channels[0]),
            nn.ReLU(True),
            nn.Linear(channels[0], channels[0]),
        )
        self.prev_transformer = PointTransformerBlock(channels[0], k)

        self.trans_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()

        for i in range(1, len(channels) - 2):
            self.trans_downs.append(
                TransitionDown(
                    in_channels=channels[i - 1],
                    out_channels=channels[i],
                    k=k,
                    sampling_ratio=sampling_ratio,
                )
            )
            self.transformers.append(PointTransformerBlock(channels[i], k))

        self.final_block = nn.Sequential(
            nn.Linear(channels[-3], channels[-2]),
            nn.ReLU(True),
            nn.Linear(channels[-2], channels[-1]),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
        Forward pass of the network

        Parameters
        ----------
        pointcloud: Variable(torch.cuda.FloatTensor)
            (B, N, 3 + input_channels) tensor
            Point cloud to run predicts on
            Each point in the point-cloud MUST
            be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        # Timers
        t_prev = Timer("prev_block")
        t_prev.tic()
        features = self.prev_block(xyz)
        t_prev.toc()

        t_prev_trs = Timer("prev_transformer")
        t_prev_trs.tic()
        features = self.prev_transformer(features, xyz)
        t_prev_trs.toc()

        t_td = Timer("transition_down")
        t_trs = Timer("transformer")
        for trans_down_layer, transformer_layer in zip(
            self.trans_downs, self.transformers
        ):
            t_td.tic()
            features, xyz = trans_down_layer(features, xyz)
            t_td.toc()

            t_trs.tic()
            features = transformer_layer(features, xyz)
            t_trs.toc()

        t_final = Timer("final_block")
        t_final.tic()
        out = self.final_block(features.mean(1))
        t_final.toc()
        return out

    def configure_optimizers(self):
        """
        SGD: momentum=0.9, weight_decay = 0.0001
        Max epoch: 200
        Initial learning rate = 0.05
        Drop 10x at epoch 120, 160
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams["optimizer.lr"],
            weight_decay=self.hparams["optimizer.weight_decay"],
            momentum=self.hparams["optimizer.momentum"],
        )
        milestones = list(map(int, self.hparams["optimizer.milestones"].split(".")))
        lr_scheduler = lr_sched.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=self.hparams["optimizer.gamma"],
        )

        return [optimizer], [lr_scheduler]