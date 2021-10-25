from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops.functions import pointops
from util import block
from model.pointnet2 import paconv


class _PointNet2SAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N0, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, Cin, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, N1, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, Cout, N1)) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        if self.npoint is None:
            self.npoint = xyz.shape[1] // 4
        new_xyz_idx = pointops.furthestsampling(xyz, self.npoint)  # (B, N1)
        new_xyz = pointops.gathering(
            xyz_trans,
            new_xyz_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  # (B, N1, 3)
        for i in range(len(self.groupers)):
            new_features, grouped_xyz, _ = self.groupers[i](xyz, new_xyz, features)
            # (B, Cin+3, N1, K), (B, 3, N1, K)
            if isinstance(self.mlps[i], paconv.SharedPAConv):
                new_features = self.mlps[i]((new_features, grouped_xyz))[0]  # (B, Cout, N1, K)
            else:
                new_features = self.mlps[i](new_features)  # (B, Cout, N1, K)
            if self.agg == 'max':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(-1)])  # (B, Cout, N1, 1)
            elif self.agg == 'sum':
                new_features = torch.sum(new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
            elif self.agg == 'avg':
                new_features = torch.mean(new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
            else:
                raise ValueError('Not implemented aggregation mode.')
            new_features = new_features.squeeze(-1)  # (B, Cout, N1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2SAModuleMSG(_PointNet2SAModuleBase):
    r"""Pointnet set abstraction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet_old before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True, use_xyz: bool = True, use_paconv: bool = False, voxel_size=None, args=None):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.use_xyz = use_xyz
        self.agg = args.get('agg', 'max')
        self.sampling = args.get('sampling', 'fps')
        self.voxel_size = voxel_size
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointops.QueryAndGroup(radius, nsample, use_xyz=use_xyz, return_idx=True)
                # if npoint is not None else pointops.GroupAll(use_xyz=use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if use_paconv:
                self.mlps.append(paconv.SharedPAConv(mlp_spec, bn=bn, config=args))
            else:
                self.mlps.append(block.SharedMLP(mlp_spec, bn=bn))


class PointNet2SAModule(PointNet2SAModuleMSG):
    r"""Pointnet set abstraction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True, use_paconv: bool = False, args=None):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, use_paconv=use_paconv, args=args)


class PointNet2SAModuleCUDA(PointNet2SAModuleMSG):
    r"""Pointnet set abstraction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet_old before the global max_pool
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True, use_paconv: bool = False, args=None):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, use_paconv=use_paconv, args=args)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N0, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, Cin, N0) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, N1, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, Cout, N1)) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        if self.npoint is None:
            self.npoint = xyz.shape[1] // 4
        new_xyz_idx = pointops.furthestsampling(xyz, self.npoint)  # (B, N1)
        new_xyz = pointops.gathering(
            xyz_trans,
            new_xyz_idx
        ).transpose(1, 2).contiguous() if self.npoint is not None else None  # (B, N1, 3)
        new_features = features
        for i in range(len(self.groupers)):
            for j in range(len(self.mlps[i])):
                _, grouped_xyz, grouped_idx = self.groupers[i](xyz, new_xyz, new_features)
                # (B, Cin+3, N1, K), (B, 3, N1, K), (B, N1, K)
                if self.use_xyz and j == 0:
                    new_features = torch.cat((xyz.permute(0, 2, 1), new_features), dim=1)
                if isinstance(self.mlps[i], paconv.SharedPAConv):
                    grouped_new_features = self.mlps[i][j]((new_features, grouped_xyz, grouped_idx))[0]  # (B, Cout, N1, K)
                else:
                    raise NotImplementedError
                if self.agg == 'max':
                    new_features = F.max_pool2d(grouped_new_features, kernel_size=[1, grouped_new_features.size(3)])  # (B, Cout, N1, 1)
                elif self.agg == 'sum':
                    new_features = torch.sum(grouped_new_features, dim=-1, keepdim=True)  # (B, Cout, N1, 1)
                else:
                    raise ValueError('Not implemented aggregation mode.')
                xyz = new_xyz
                new_features = new_features.squeeze(-1).contiguous()  # (B, Cout, N1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2FPModule(nn.Module):
    r"""Propagates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, mlp: List[int], bn: bool = True,  use_paconv=False, args=None):
        super().__init__()
        self.use_paconv = use_paconv
        if self.use_paconv:
            self.mlp = paconv.SharedPAConv(mlp, bn=bn, config=args)
        else:
            self.mlp = block.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated
        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointops.interpolation(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        return self.mlp(new_features.unsqueeze(-1)).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = torch.randn(2, 9, 3, requires_grad=True).cuda()
    xyz_feats = torch.randn(2, 9, 6, requires_grad=True).cuda()

    test_module = PointNet2SAModuleMSG(npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]])
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    # test_module = PointNet2FPModule(mlp=[6, 6])
    # test_module.cuda()
    # from torch.autograd import gradcheck
    # inputs = (xyz, xyz, None, xyz_feats)
    # test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    # print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
