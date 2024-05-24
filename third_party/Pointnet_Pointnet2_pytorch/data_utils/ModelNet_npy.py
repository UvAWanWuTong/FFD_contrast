from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from math import  *
import  tqdm
import pickle
import re

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'meta/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def VaryPoint(data, axis, degree):
    xyzArray = {
        'X': np.array([[1, 0, 0],
                  [0, cos(radians(degree)), -sin(radians(degree))],
                  [0, sin(radians(degree)), cos(radians(degree))]]),
        'Y': np.array([[cos(radians(degree)), 0, sin(radians(degree))],
                  [0, 1, 0],
                  [-sin(radians(degree)), 0, cos(radians(degree))]]),
        'Z': np.array([[cos(radians(degree)), -sin(radians(degree)), 0],
                  [sin(radians(degree)), cos(radians(degree)), 0],
                  [0, 0, 1]])}
    newData = np.dot(data, xyzArray[axis])
    return newData


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point





class Deform_ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=1024,
                 split='train',
                 data_augmentation=True,
                 deform=True):

        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []

        if deform:
            with open(os.path.join(root, '{}_deform.txt'.format(self.split)), 'r') as f:
                for line in f:
                    self.fns.append(line.strip())

            self.cat = {}
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'meta/modelnet_id.txt'), 'r') as f:
                for line in f:
                    ls = line.strip().split()
                    self.cat[ls[0]] = int(ls[1])
        else:
            with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
                for line in f:
                    self.fns.append(line.strip())

            self.cat = {}
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'meta/modelnet_id.txt'), 'r') as f:
                for line in f:
                    ls = line.strip().split()
                    self.cat[ls[0]] = int(ls[1])


        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        npy_data = np.load(os.path.join(self.root, fn))
        # choice = np.random.choice(len(npy_data), self.npoints, replace=True)
        if np.isnan(npy_data[0][0]):
            return  None,cls

        xyz_rotate_x = VaryPoint(data=npy_data, axis='X', degree=270)
        xyz_rotate_z= VaryPoint(data=xyz_rotate_x, axis='Z', degree=180)
        point_set = pc_normalize(xyz_rotate_z[0:self.npoints, :])

        # if self.data_augmentation:
        #     theta = np.random.uniform(0, np.pi * 2)
        #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        #     point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
        #     point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls[0]

class ModelNetDataset_npy(data.Dataset):
    def __init__(self, root, args, deform,split='train', process_data=False,):

            self.root = root
            self.npoints = args.num_points
            self.process_data = process_data
            self.uniform = args.use_uniform_sample
            self.use_normals = args.use_normals
            self.num_category = args.num_category

            if self.num_category == 10:
                self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
            else:
                self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

            self.cat = [line.rstrip() for line in open(self.catfile)]
            self.classes = dict(zip(self.cat, range(len(self.cat))))

            shape_ids = {}

            suffix = '_deform.txt' if deform else '.txt'

            if self.num_category == 10:
                base_name_train = 'modelnet10_train' + suffix
                base_name_test = 'modelnet10_test'  + suffix
                shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, base_name_train))]
                shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, base_name_test))]
            else:

                base_name_train = 'modelnet40_train'+ suffix
                base_name_test = 'modelnet40_test'  + suffix

                shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, base_name_train))]
                shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, base_name_test))]

            assert (split == 'train' or split == 'test')
            shape_names = [re.findall(r'^(.*?)_\d', x)[0] for x in shape_ids[split]]
            self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.npy') for i
                             in range(len(shape_ids[split]))]
            print('The size of %s data is %d' % (split, len(self.datapath)))

            if self.uniform:
                self.save_path = os.path.join(root,
                                              'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
            else:
                self.save_path = os.path.join(root,
                                              'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

            if self.process_data:
                if not os.path.exists(self.save_path):
                    print('Processing data %s (only running in the first time)...' % self.save_path)
                    self.list_of_points = [None] * len(self.datapath)
                    self.list_of_labels = [None] * len(self.datapath)

                    for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                        fn = self.datapath[index]
                        cls = self.classes[self.datapath[index][0]]
                        cls = np.array([cls]).astype(np.int32)
                        # point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
                        point_set = np.load(fn[1]).astype(np.float32)
                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                        else:
                            point_set = point_set[0:self.npoints, :]

                        self.list_of_points[index] = point_set
                        self.list_of_labels[index] = cls

                    with open(self.save_path, 'wb') as f:
                        pickle.dump([self.list_of_points, self.list_of_labels], f)
                else:
                    print('Load processed data from %s...' % self.save_path)
                    with open(self.save_path, 'rb') as f:
                        self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
                return len(self.datapath)

    def _get_item(self, index):
                if self.process_data:
                    point_set, label = self.list_of_points[index], self.list_of_labels[index]
                else:
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    label = np.array([cls]).astype(np.int32)
                    point_set = np.load(fn[1]).astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
                if not self.use_normals:
                    point_set = point_set[:, 0:3]

                return point_set, label[0]

    def __getitem__(self, index):
                return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = Deform_ModelNetDataset('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)