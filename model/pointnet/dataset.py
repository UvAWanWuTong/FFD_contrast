from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
import pygem
from pygem import FFD
from plyfile import PlyData, PlyElement


def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../ModelNet40/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class Contrastive_ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=False,
                 FFD = True):
        self.num_points = 6
        self.ffd = FFD
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../ModelNet40/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def FreeFormDeformation(self,point_set):
        def random_move(ffd):
            # randomly move the control points of  the ffd
            ffd = ffd
            point = np.random.randint(0, 3, size=[3])
            ffd.array_mu_x[point[0], point[1], point[2]] = np.random.uniform(0.5, 1.5)
            ffd.array_mu_z[point[0], point[1], point[2]] = np.random.uniform(0.5, 1.5)
            ffd.array_mu_z[point[0], point[1], point[2]] = np.random.uniform(0.5, 1.5)
            return ffd

        # initialize the 27 control points with box length of 2
        ffd = FFD([3, 3, 3])
        ffd.box_length = [2, 2, 2]
        for i in range(self.num_points):
             ffd = random_move(ffd)

        deformed_points = ffd(np.asarray(point_set)+1)

        return deformed_points-1
    def NormalDataAugmentation(self,point_set):
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
        point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
        return point_set


    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        point_set1 = point_set.copy()
        point_set2 = point_set.copy()


        if self.data_augmentation:
            point_set1 = self.NormalDataAugmentation(point_set)
            point_set2 = self.NormalDataAugmentation(point_set)
        if self.ffd:
            point_set1 = self.FreeFormDeformation(point_set)
            point_set2 = self.FreeFormDeformation(point_set)

        point_set1 = point_set1 - np.expand_dims(np.mean(point_set1, axis=0), 0)  # center
        point_set2 = point_set2 - np.expand_dims(np.mean(point_set2, axis=0), 0)  # center

        point_set1 = torch.from_numpy(point_set1.astype(np.float32))
        point_set2 = torch.from_numpy(point_set2.astype(np.float32))



        return (point_set1 , point_set2)


    def __len__(self):
        return len(self.fns)


class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../ModelNet40/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)



if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]


    if dataset == 'modelnet':
        Contrastive_ModelNetDataset(datapath)
        d= Contrastive_ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])
    dataloader = torch.utils.data.DataLoader(d,batch_size=8,shuffle=True,num_workers=4,collate_fn=collate_pair_fn,drop_last=True)
    data_loader_iter = iter(dataloader)
    input_dict = next(data_loader_iter)
    print(input_dict)
    for i, data in enumerate(dataloader,0):
            input_dict = data




