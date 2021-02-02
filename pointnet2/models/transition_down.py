import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pointnet2_ops._ext as _ext

from scipy.spatial import distance
   
def kNN(self, p, idx, k):
    '''
    inputs
        p: (n, 3) shaped torch Tensor
        idx: (n * sampling_ratio, ) shaped torch Tensor
        k: int
    
    outputs
        neighbors: (n * sampling_ratio, k) shaped numpy array
                   Each row is a point in x and each column is an index of a neighboring point.
    '''
    dist = distance.squareform(distance.pdist(p.cpu().detach().numpy()))
    closest = np.argsort(dist, axis=1)
    neighbors = closest[idx.cpu().detach().numpy(), 1:k+1]
    
    return neighbors

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, k, sampling_ratio):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.sampling_ratio = sampling_ratio
        self.mlp_layer = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x, p1):
        '''
        inputs
            x: (n, in_channels) shaped torch Tensor (A set of feature vectors)
            p1: (n, 3) shaped torch Tensor (3D coordinates)
        
        outputs
            y: (n * sampling_ratio, out_channels) shaped torch Tensor
            p2: (n * sampling_ratio, 3) shaped torch Tensor
        '''
        n = p1.shape[0]
        sampled_n = int(n * self.sampling_ratio)
        
        # 1: Furthest Point Sampling
        p1 = p1.unsqueeze(0)
        sampled_index = _ext.furthest_point_sampling(p1, sampled_n)
        sampled_index = sampled_index.squeeze(0).long()
        p1 = p1.squeeze(0)
        p2 = p1[sampled_index]
        
        # 2: kNN & MLP
        neighbors = kNN(p1, sampled_index, self.k) # "neighbors" is a (sampled_n, k) shaped numpy array.
        
        # 2-1: Apply MLP onto each feature
        mlp_x = self.mlp_layer(x) # "mlp_x" is a (n, out_channels) shaped torch Tensor.
        
        # 2-2: Extract features based on neighbors
        features = []
        for i in range(sampled_n):
            idx = neighbors[i, :]
            feature = mlp_x[idx].unsqueeze(0)
            features.append(feature)
        features = torch.cat(features) # "features" is a (sampled_n, k, out_channels) shaped torch Tensor.
        
        # 3: Local Max Pooling
        y = self.local_max_pooling(features)
        
        return y, p2
    
    def local_max_pooling(self, features):
        '''
        inputs
            features: (n * sampling_ratio, k, out_channels) shaped torch Tensor
        
        outputs
            pooled_features: (n * sampling_ratio, out_channles) shaped torch Tensor
        '''
        pooled_features = []
        for i in range(features.shape[0]):
            feature = features[i]
            pooled_feature = torch.max(feature, dim=0)[0].unsqueeze(0)
            pooled_features.append(pooled_feature)
        pooled_features = torch.cat(pooled_features) # "pooled_features" is a (n * sampling_ratio, out_channels) shaped torch Tensor.
        
        return pooled_features