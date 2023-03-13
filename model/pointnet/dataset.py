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


class ModelNetDataset(data.Dataset):
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


        point_set1 = torch.from_numpy(point_set1.astype(np.float32))
        point_set2 = torch.from_numpy(point_set2.astype(np.float32))

        matching_idx = np.column_stack((np.arange(point_set1.size()[0]), np.arange(point_set2.size()[0])))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return (point_set1 , point_set2, matching_idx, cls)


    def __len__(self):
        return len(self.fns)


def default_collate_pair_fn(list_data):
    xyz0, xyz1, matching_inds, cls = list(
        zip(*list_data))
    xyz_batch0, coords_batch0, feats_batch0 = [], [], []
    xyz_batch1, coords_batch1, feats_batch1 = [], [], []
    matching_inds_batch, class_batch, len_batch = [], [], []

    batch_id = 0
    curr_start_inds = np.zeros((1, 2))
    for batch_id, _ in enumerate(xyz0):

        N0 = xyz0[batch_id].shape[0]
        N1 = xyz0[batch_id].shape[0]

        # Move batchids to the beginning
        xyz_batch0.append(xyz0[batch_id])
        # coords_batch : batch id, x,y,z
        xyz_batch1.append(xyz1[batch_id])


        # in case 0 matching
        if len(matching_inds[batch_id]) == 0:
            matching_inds[batch_id].extend([0, 0])

        matching_inds_batch.append(
            torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
        class_batch.append(cls[batch_id])

        len_batch.append([N0, N1])



        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    # Concatenate all lists
    xyz_batch0 = torch.cat(xyz_batch0, 0).float()
    xyz_batch1 = torch.cat(xyz_batch1, 0).float()
    class_batch = torch.cat(class_batch, 0).float()
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
    return {
        'pcd0': xyz_batch0,
        'pcd1': xyz_batch1,
        'correspondences': matching_inds_batch,
        'class':class_batch,
        'len_batch': len_batch,
    }


if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]


    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d= ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])
    collate_pair_fn = default_collate_pair_fn
    dataloader = torch.utils.data.DataLoader(d,batch_size=8,shuffle=True,num_workers=4,collate_fn=collate_pair_fn,drop_last=True)
    data_loader_iter = iter(dataloader)
    input_dict = next(data_loader_iter)
    print(input_dict)
    for i, data in enumerate(dataloader,0):
            input_dict = data




