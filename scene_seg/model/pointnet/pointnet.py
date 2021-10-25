import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3D(nn.Module):
    def __init__(self, c):
        super(STN3D, self).__init__()
        self.c = c
        self.conv1 = nn.Conv1d(self.c, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.mp = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.c*self.c)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp(x)
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.c).view(1, -1).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.c, self.c)
        return x


class PointNetFeat(nn.Module):
    def __init__(self, c=3, global_feat=True):
        super(PointNetFeat, self).__init__()
        self.global_feat = global_feat
        self.stn1 = STN3D(c)
        self.conv1 = nn.Conv1d(c, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.stn2 = STN3D(64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.mp = nn.AdaptiveMaxPool1d(1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        stn1 = self.stn1(x)
        x = torch.bmm(stn1, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        stn2 = self.stn2(x)
        x_tmp = torch.bmm(stn2, x)
        x = F.relu(self.bn3(self.conv3(x_tmp)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.mp(x)
        x = x.view(-1, 1024)

        if not self.global_feat:
            x = x.view(-1, 1024, 1).repeat(1, 1, x_tmp.size()[2])
            x = torch.cat([x_tmp, x], 1)
        return x


class PointNetCls(nn.Module):
    def __init__(self, c=3, k=40, dropout=0.3, sync_bn=False):
        super(PointNetCls, self).__init__()
        self.feat = PointNetFeat(c, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Segmentation with 9 channels input XYZ, RGB and normalized location to the room (from 0 to 1), with STN3D on input and feature
class PointNetSeg(nn.Module):
    def __init__(self, c=9, k=13, sync_bn=False):
        super(PointNetSeg, self).__init__()
        self.feat = PointNetFeat(c, global_feat=False)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, 128, 1)
        self.conv5 = nn.Conv1d(128, k, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    sim_data = torch.rand(16, 2048, 3)

    trans = STN3D(c=3)
    out = trans(sim_data.transpose(1, 2))
    print('stn', out.size())

    point_feat = PointNetFeat(global_feat=True)
    out = point_feat(sim_data.transpose(1, 2))
    print('global feat', out.size())

    point_feat = PointNetFeat(global_feat=False)
    out = point_feat(sim_data.transpose(1, 2))
    print('point feat', out.size())

    cls = PointNetCls(c=3, k=40)
    out = cls(sim_data)
    print('class', out.size())

    sim_data = torch.rand(16, 2048, 9)
    seg = PointNetSeg(c=9, k=13)
    out = seg(sim_data)
    print('seg', out.size())
