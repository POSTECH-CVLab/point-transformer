"""
Embed PAConv into PointNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.PAConv_util import get_scorenet_input, knn, feat_trans_pointnet, ScoreNet
from cuda_lib.functional import assign_score_withk_halfkernel as assemble_pointnet


class PAConv(nn.Module):
    def __init__(self, args):
        super(PAConv, self).__init__()
        self.args = args
        self.k = args.get('k_neighbors', 20)
        self.calc_scores = args.get('calc_scores', 'softmax')

        self.m2, self.m3, self.m4 = args.get('num_matrices', [8, 8, 8])
        self.scorenet2 = ScoreNet(6, self.m2, hidden_unit=[16])
        self.scorenet3 = ScoreNet(6, self.m3, hidden_unit=[16])
        self.scorenet4 = ScoreNet(6, self.m4, hidden_unit=[16])

        i2 = 64  # channel dim of output_1st and input_2nd
        o2 = i3 = 64  # channel dim of output_2st and input_3rd
        o3 = i4 = 64  # channel dim of output_3rd and input_4th
        o4 = 128  # channel dim of output_4th and input_5th

        tensor2 = nn.init.kaiming_normal_(torch.empty(self.m2, i2, o2), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i2, self.m2 * o2)
        tensor3 = nn.init.kaiming_normal_(torch.empty(self.m3, i3, o3), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i3, self.m3 * o3)
        tensor4 = nn.init.kaiming_normal_(torch.empty(self.m4, i4, o4), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i4, self.m4 * o4)

        # convolutional weight matrices in Weight Bank:
        self.matrice2 = nn.Parameter(tensor2, requires_grad=True)
        self.matrice3 = nn.Parameter(tensor3, requires_grad=True)
        self.matrice4 = nn.Parameter(tensor4, requires_grad=True)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 40)

    def forward(self, x, label=None, criterion=None):
        batch_size = x.size(0)
        idx, _ = knn(x, k=self.k)  # get the idx of knn in 3D space : b,n,k
        xyz = get_scorenet_input(x, k=self.k, idx=idx)  # ScoreNet input: 3D coord difference : b,6,n,k

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        ##################
        # replace the intermediate 3 MLP layers with PAConv:
        """CUDA implementation of PAConv: (presented in the supplementary material of the paper)"""
        """feature transformation:"""
        x = feat_trans_pointnet(point_input=x, kernel=self.matrice2, m=self.m2)  # b,n,m1,o1
        score2 = self.scorenet2(xyz, calc_scores=self.calc_scores, bias=0)
        """assemble with scores:"""
        x = assemble_pointnet(score=score2, point_input=x, knn_idx=idx, aggregate='sum')   # b,o1,n
        x = F.relu(self.bn2(x))

        x = feat_trans_pointnet(point_input=x, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_pointnet(score=score3, point_input=x, knn_idx=idx, aggregate='sum')
        x = F.relu(self.bn3(x))

        x = feat_trans_pointnet(point_input=x, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_pointnet(score=score4, point_input=x, knn_idx=idx, aggregate='sum')
        x = F.relu(self.bn4(x))
        ##################
        x = self.conv5(x)
        x = F.relu(self.bn5(x))

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        if criterion is not None:
            return x, criterion(x, label)
        else:
            return x
