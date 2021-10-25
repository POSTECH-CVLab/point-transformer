from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from util.paconv_util import weight_init, assign_score, get_ed, assign_kernel_withoutk
from lib.paconv_lib.functional import assign_score_withk as assign_score_cuda


class ScoreNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], last_bn=False, temp=1):
        super(ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()
        self.temp = temp

        hidden_unit = list() if hidden_unit is None else copy.deepcopy(hidden_unit)
        hidden_unit.append(out_channel)
        hidden_unit.insert(0, in_channel)

        for i in range(1, len(hidden_unit)):  # from 1st hidden to next hidden to last hidden
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1,
                                                   bias=False if i < len(hidden_unit) - 1 else not last_bn))
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))

    def forward(self, xyz, score_norm='softmax'):
        # xyz : B*3*N*K
        B, _, N, K = xyz.size()
        scores = xyz

        for i, conv in enumerate(self.mlp_convs_hidden):
            if i < len(self.mlp_convs_hidden) - 1:
                scores = F.relu(self.mlp_bns_hidden[i](conv(scores)))
            else:  # if the output layer, no ReLU
                scores = conv(scores)
                if self.last_bn:
                    scores = self.mlp_bns_hidden[i](scores)
        if score_norm == 'softmax':
            scores = F.softmax(scores/self.temp, dim=1)  # + 0.5  # B*m*N*K
        elif score_norm == 'sigmoid':
            scores = torch.sigmoid(scores/self.temp)  # + 0.5  # B*m*N*K
        elif score_norm is None:
            scores = scores
        else:
            raise ValueError('Not Implemented!')

        scores = scores.permute(0, 2, 3, 1)  # B*N*K*m

        return scores


class PAConv(nn.Module):

    def __init__(self, input_dim, output_dim, bn, activation, config):
        super().__init__()
        self.score_input = config.get('score_input', 'identity')
        self.score_norm = config.get('score_norm', 'softmax')
        self.temp = config.get('temp', 1)
        self.init = config.get('init', 'kaiming')
        self.hidden = config.get('hidden', [16])
        self.m = config.get('m', 8)
        self.kernel_input = config.get('kernel_input', 'neighbor')
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.bn = nn.BatchNorm2d(output_dim, momentum=0.1) if bn else None
        self.activation = activation

        if self.kernel_input == 'identity':
            self.kernel_mul = 1
        elif self.kernel_input == 'neighbor':
            self.kernel_mul = 2
        else:
            raise ValueError()

        if self.score_input == 'identity':
            self.scorenet_input_dim = 3
        elif self.score_input == 'neighbor':
            self.scorenet_input_dim = 6
        elif self.score_input == 'ed7':
            self.scorenet_input_dim = 7
        elif self.score_input == 'ed':
            self.scorenet_input_dim = 10
        else: raise ValueError()

        if self.init == "kaiming":
            _init = nn.init.kaiming_normal_
        elif self.init == "xavier":
            _init = nn.init.xavier_normal_
        else:
            raise ValueError('Not implemented!')

        self.scorenet = ScoreNet(self.scorenet_input_dim, self.m, hidden_unit=self.hidden, last_bn=False, temp=self.temp)

        tensor1 = _init(torch.empty(self.m, input_dim * self.kernel_mul, output_dim)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(input_dim * self.kernel_mul, self.m * output_dim)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)

        for m in self.modules():
            weight_init(m)

    def forward(self, args):
        r"""
            Parameters
            ----------
            in_feat : torch.Tensor
                (B, C, N1, K) tensor of the descriptors of the the features
            grouped_xyz : torch.Tensor
                (B, 3, N1, K) tensor of the descriptors of the the features
            Returns
            -------
            out_feat : torch.Tensor
                (B, C, N1, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
            """

        in_feat, grouped_xyz = args
        B, _, N1, K = in_feat.size()
        center_xyz = grouped_xyz[..., :1].repeat(1, 1, 1, K)
        grouped_xyz_diff = grouped_xyz - center_xyz  # b,3,n1,k
        if self.kernel_input == 'neighbor':
            in_feat_c = in_feat[..., :1].repeat(1, 1, 1, K)
            in_feat_diff = in_feat - in_feat_c
            in_feat = torch.cat((in_feat_diff, in_feat), dim=1)

        ed = get_ed(center_xyz.permute(0, 2, 3, 1).reshape(B * N1 * K, -1),
                    grouped_xyz.permute(0, 2, 3, 1).reshape(B * N1 * K, -1)).reshape(B, 1, N1, K)
        if self.score_input == 'neighbor':
            xyz = torch.cat((grouped_xyz_diff, grouped_xyz), dim=1)
        elif self.score_input == 'identity':
            xyz = grouped_xyz_diff
        elif self.score_input == 'ed7':
            xyz = torch.cat((center_xyz, grouped_xyz_diff, ed), dim=1)
        elif self.score_input == 'ed10':
            xyz = torch.cat((center_xyz, grouped_xyz, grouped_xyz_diff, ed), dim=1)
        else:
            raise NotImplementedError

        scores = self.scorenet(xyz, score_norm=self.score_norm)  # b,n,k,m
        out_feat = torch.matmul(in_feat.permute(0, 2, 3, 1), self.weightbank).view(B, N1, K, self.m, -1)  # b,n1,k,m,cout
        out_feat = assign_score(score=scores, point_input=out_feat)  # b,n,k,o1,
        out_feat = out_feat.permute(0, 3, 1, 2)  # b,o1,n,k

        if self.bn is not None:
            out_feat = self.bn(out_feat)
        if self.activation is not None:
            out_feat = self.activation(out_feat)

        return out_feat, grouped_xyz  # b,o1,n,k   b,3,n1,k

    def __repr__(self):
        return 'PAConv(in_feat: {:d}, out_feat: {:d}, m: {:d}, hidden: {}, scorenet_input: {}, kernel_size: {})'.\
            format(self.input_dim, self.output_dim, self.m, self.hidden, self.scorenet_input_dim, self.weightbank.shape)


class PAConvCUDA(PAConv):

    def __init__(self, input_dim, output_dim, bn, activation, config):
        super(PAConvCUDA, self).__init__(input_dim, output_dim, bn, activation, config)

    def forward(self, args):

        r"""
            Parameters
            ----------
            in_feat : torch.Tensor
                (B, C, N0) tensor of the descriptors of the the features
            grouped_xyz : torch.Tensor
                (B, 3, N1, K) tensor of the descriptors of the the features
            grouped_idx : torch.Tensor
                (B, N1, K) tensor of the descriptors of the the features
            Returns
            -------
            out_feat : torch.Tensor
                (B, C, N1) tensor of the new_features descriptors
            new_xyz : torch.Tensor
                (B, N1, 3) tensor of the new features' xyz
            """
        in_feat, grouped_xyz, grouped_idx = args
        B, Cin, N0 = in_feat.size()
        _, _, N1, K = grouped_xyz.size()
        center_xyz = grouped_xyz[..., :1].repeat(1, 1, 1, K)
        grouped_xyz_diff = grouped_xyz - center_xyz  # [B, 3, N1, K]

        ed = get_ed(center_xyz.permute(0, 2, 3, 1).reshape(B * N1 * K, -1),
                    grouped_xyz.permute(0, 2, 3, 1).reshape(B * N1 * K, -1)).reshape(B, 1, N1, K)

        if self.score_input == 'neighbor':
            xyz = torch.cat((grouped_xyz_diff, grouped_xyz), dim=1)
        elif self.score_input == 'identity':
            xyz = grouped_xyz_diff
        elif self.score_input == 'ed7':
            xyz = torch.cat((center_xyz, grouped_xyz_diff, ed), dim=1)
        elif self.score_input == 'ed':
            xyz = torch.cat((center_xyz, grouped_xyz, grouped_xyz_diff, ed), dim=1)
        else:
            raise NotImplementedError

        scores = self.scorenet(xyz, score_norm=self.score_norm)  # b,n1,k,m
        kernel_feat, half_kernel_feat = assign_kernel_withoutk(in_feat, self.weightbank, self.m)
        out_feat = assign_score_cuda(scores, kernel_feat, half_kernel_feat, grouped_idx, aggregate='sum')  # b,o1,n1,k
        if self.bn is not None:
            out_feat = self.bn(out_feat)
        if self.activation is not None:
            out_feat = self.activation(out_feat)

        return out_feat, grouped_xyz, grouped_idx  # b,o1,n,k

    def __repr__(self):
        return 'PAConvCUDA(in_feat: {:d}, out_feat: {:d}, m: {:d}, hidden: {}, scorenet_input: {}, kernel_size: {})'.\
            format(self.input_dim, self.output_dim, self.m, self.hidden, self.scorenet_input_dim, self.weightbank.shape)


class SharedPAConv(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            config,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = "",
    ):
        super().__init__()

        for i in range(len(args) - 1):
            if config.get('cuda', False):
                self.add_module(
                    name + 'layer{}'.format(i),
                    PAConvCUDA(
                        args[i],
                        args[i + 1],
                        bn=(not first or not preact or (i != 0)) and bn,
                        activation=activation
                        if (not first or not preact or (i != 0)) else None,
                        config=config,
                    )
                )
            else:
                self.add_module(
                    name + 'layer{}'.format(i),
                    PAConv(
                        args[i],
                        args[i + 1],
                        bn=(not first or not preact or (i != 0)) and bn,
                        activation=activation
                        if (not first or not preact or (i != 0)) else None,
                        config=config,
                    )
            )
