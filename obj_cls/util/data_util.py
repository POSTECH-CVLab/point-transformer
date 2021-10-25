import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def load_data(partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob('./data/modelnet40_ply_hdf5_2048/ply_data_%s*.h5' % partition):
        f = h5py.File(h5_name, mode='r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', pt_norm=False):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.pt_norm = pt_norm

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            if self.pt_norm:
                pointcloud = pc_normalize(pointcloud)
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)  # shuffle the order of pts
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
