from collections import namedtuple

import torch
import torch.nn as nn

from model.pointnet2.pointnet2_modules import PointNet2SAModule, PointNet2SAModuleMSG, PointNet2FPModule
from util import block


class PointNet2SSGSeg(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        k: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        c: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, c=3, k=13, use_xyz=True, args=None):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(PointNet2SAModule(npoint=1024, nsample=32, mlp=[c, 32, 32, 64], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(npoint=256, nsample=32, mlp=[64, 64, 64, 128], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(npoint=64, nsample=32, mlp=[128, 128, 128, 256], use_xyz=use_xyz))
        self.SA_modules.append(PointNet2SAModule(npoint=16, nsample=32, mlp=[256, 256, 256, 512], use_xyz=use_xyz))
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNet2FPModule(mlp=[128 + c, 128, 128, 128]))
        self.FP_modules.append(PointNet2FPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointNet2FPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointNet2FPModule(mlp=[512 + 256, 256, 256]))
        self.FC_layer = nn.Sequential(block.Conv2d(128, 128, bn=True), nn.Dropout(), block.Conv2d(128, k, activation=None))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
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
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        # return self.FC_layer(l_features[0])
        return self.FC_layer(l_features[0].unsqueeze(-1)).squeeze(-1)


class PointNet2MSGSeg(PointNet2SSGSeg):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        k: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        c: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, k, c=6, use_xyz=True):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        c_in = c
        self.SA_modules.append(PointNet2SAModuleMSG(npoint=1024, radii=[0.05, 0.1], nsamples=[16, 32], mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]], use_xyz=use_xyz ))
        c_out_0 = 32 + 64
        c_in = c_out_0
        self.SA_modules.append(PointNet2SAModuleMSG(npoint=256, radii=[0.1, 0.2], nsamples=[16, 32], mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]], use_xyz=use_xyz))
        c_out_1 = 128 + 128
        c_in = c_out_1
        self.SA_modules.append(PointNet2SAModuleMSG(npoint=64, radii=[0.2, 0.4], nsamples=[16, 32], mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]], use_xyz=use_xyz))
        c_out_2 = 256 + 256
        c_in = c_out_2
        self.SA_modules.append(PointNet2SAModuleMSG(npoint=16, radii=[0.4, 0.8], nsamples=[16, 32], mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]], use_xyz=use_xyz))
        c_out_3 = 512 + 512
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNet2FPModule(mlp=[256 + c, 128, 128]))
        self.FP_modules.append(PointNet2FPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointNet2FPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointNet2FPModule(mlp=[c_out_3 + c_out_2, 512, 512]))
        self.FC_layer = nn.Sequential(block.Conv2d(128, 128, bn=True), nn.Dropout(), block.Conv2d(128, k, activation=None))


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, data, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            preds = model(inputs)
            loss = criterion(preds, labels)
            _, classes = torch.max(preds, 1)
            acc = (classes == labels).float().sum() / labels.numel()
            return ModelReturn(preds, loss, {"acc": acc.item(), 'loss': loss.item()})
    return model_fn


if __name__ == "__main__":
    import torch.optim as optim
    B, N, C, K = 2, 4096, 3, 13
    inputs = torch.randn(B, N, 6)#.cuda()
    labels = torch.randint(0, 3, (B, N))#.cuda()

    model = PointNet2SSGSeg(c=C, k=K)#.cuda()
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    print("Testing SSGCls with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()

    model = PointNet2SSGSeg(c=C, k=K, use_xyz=False).cuda()
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    print("Testing SSGCls without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()

    model = PointNet2MSGSeg(c=C, k=K).cuda()
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    print("Testing MSGCls with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()

    model = PointNet2MSGSeg(c=C, k=K, use_xyz=False).cuda()
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
    print("Testing MSGCls without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.item())
        optimizer.step()

