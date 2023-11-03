import os
import shutil

import torch
import yaml
import numpy as np
import  torch
from torch import nn

def clean_dir(directory):
        shutil.rmtree(os.path.join(directory))
        os.makedirs(directory)



def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


import shutil

import torch
import yaml
import numpy as np

def clean_dir(directory):
        shutil.rmtree(os.path.join(directory))
        os.makedirs(directory)



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', file_dir='checkpoints', save_deform=None):
    save_checkpoint_dir = os.path.join(file_dir, 'checkpoints')
    save_best_dir = os.path.join(file_dir, 'best')
    save_deform_net_dir = os.path.join(file_dir, 'deform')

    if not os.path.exists(save_checkpoint_dir):
        os.makedirs(save_checkpoint_dir)

    if not os.path.exists(save_best_dir):
        os.makedirs(save_best_dir)

    if not os.path.exists(save_deform_net_dir):
        os.makedirs(save_deform_net_dir)

    # Optionally save the deformnet checkpoint
    if save_deform:
        deform_filename = filename  # You can modify this if needed
        torch.save(state, os.path.join(save_deform_net_dir, deform_filename))
        return

    # Save the main model checkpoint
    torch.save(state, os.path.join(save_checkpoint_dir, filename))

    # Check if it's the best model
    if is_best:
        torch.save(state, os.path.join(save_best_dir, 'best_model.pth.tar'))





def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# def chamfer_distance(pointset1,pointset2):
#
#     source_cloud = torch.randn(1, 100, 3).cuda()
#     target_cloud = torch.randn(1, 50, 3).cuda()
#
#
#
#     return source_cloud ,   target_cloud
#
#




def np_to_tensor(x):

    return torch.from_numpy(x.astype(np.float32))


def normalize_pointcloud_tensor(x):

    return nn.functional.normalize(x,p=2.0,dim=1)


def chamfer_distance(cd,xyz1,xyz2):
    return cd(xyz1.cpu() , xyz2.cpu(),bidirectional=True)

def np_to_tensor(x):

    return torch.from_numpy(x.astype(np.float32))




