import os
import h5py
import numpy as np
import torch

from torch.utils.data import Dataset


def make_dataset(split='train', data_root=None, data_list=None):
    if not os.path.isfile(data_list):
        raise (RuntimeError("Point list file do not exist: " + data_list + "\n"))
    point_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    for line in list_read:
        point_list.append(os.path.join(data_root, line.strip()))
    return point_list


class PointData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, 
                 num_point=None, random_index=False, norm_as_feat=True, fea_dim=6):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.num_point = num_point
        self.random_index = random_index
        self.norm_as_feat = norm_as_feat
        self.fea_dim = fea_dim

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_path = self.data_list[index]
        f = h5py.File(data_path, 'r')
        data = f['data'][:]
        if self.split is 'test':
            label = 255  # place holder
        else:
            label = f['label'][:]
        f.close()
        if self.num_point is None:
            self.num_point = data.shape[0]
        idxs = np.arange(data.shape[0])
        if self.random_index:
            np.random.shuffle(idxs)
        idxs = idxs[0: self.num_point]
        data = data[idxs, :]
        if label.size != 1:  # seg data
            label = label[idxs]
        if self.transform is not None:
            data, label = self.transform(data, label)

        if self.fea_dim == 3:
            points = data[:, :6]
        elif self.fea_dim == 4:
            points = np.concatenate((data[:, :6], data[:, 2:3]), axis=-1)
        elif self.fea_dim == 5:
            points = np.concatenate((data[:, :6], data[:, 2:3], torch.ones((self.num_point, 1)).to(data.device)), axis=-1)
        elif self.fea_dim == 6:
            points = data

        return points, label


if __name__ == '__main__':
    data_root = '/mnt/sda1/hszhao/dataset/3d/s3dis'
    data_list = '/mnt/sda1/hszhao/dataset/3d/s3dis/list/train12346.txt'
    point_data = PointData('train', data_root, data_list)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
