from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from model.pointnet.dataset import ModelNetDataset
import numpy as np
from model.pointnet.model import PointNetCls, feature_transform_regularizer
from utils.criterion import  NCESoftmaxLoss

import torch.nn.functional as F
from tqdm import tqdm
import wandb


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='modelnet40', help="modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    # collate_fn=default_collate_pair_fn
)

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize



# wandb.init(project="pointnet",
#            name="pointnet-classification",
#            config={
#                "architecture":"pointnet-classification",
#                "epochs": opt.nepoch,
#                "dataset":'ModelNet40'
#            }
#            )
#
# print('Iinitialization of wandb complete\n')
#

cur_device = torch.cuda.current_device()

for epoch in range(opt.nepoch):
    """ contrastive learning """
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        # input_dict = data  # points: 32 * 25000 *3  , target: 32 * 1 （batch size = 32）
        # pcd0 = input_dict['pcd0']
        # pcd1 = input_dict['pcd0']
        pcd0,pcd1,pos_pairs,cls = data

        # N0, N1 = input_dict['pcd0'].shape[0], input_dict['pcd1'].shape[0]

        # pos_pairs = input_dict['correspondences'].cuda()
        # points = points.transpose(2, 1) # 3 * 2500
        points1, points2 = pcd0.cuda(), pcd1.cuda()
        points1 = points1.transpose(2, 1)
        points2 = points2.transpose(2, 1)


        optimizer.zero_grad()
        classifier = classifier.train()
        F0,_,_ = classifier(points1)
        F1,_,_ = classifier(points2)
        F1,F0

        # random sample
        sampled_inds = np.random.choice(points1.size()[2], opt.num_points, replace=False)
        q = F0[sampled_inds]
        k = F1[sampled_inds]


        criterion = NCESoftmaxLoss().cuda()
        loss = criterion(out, labels)

        #

        #
        # count = pos_pairs[0][:,0].cuda()
        # uniform = torch.distributions.Uniform(0, 1).sample([len(count)]).cuda()
        # cums = torch.cat([torch.tensor([0], device=cur_device), torch.cumsum(count, dim=0)[0:-1]], dim=0)

#         q = F0[q_unique.long()]
#         k = F1[k_sel.long()]
#
#         q_unique, count = pos_pairs[:, 0].unique(return_counts=True)
#
#         loss =
#         loss = F.nll_loss(pred, target)
#         if opt.feature_transform:
#             loss += feature_transform_regularizer(trans_feat) * 0.001
#         loss.backward()
#         optimizer.step()
#         pred_choice = pred.data.max(1)[1]
#         correct = pred_choice.eq(target.data).cpu().sum()
#         wandb.log({"train acc": correct.item() / float(opt.batchSize * 2500), "train loss": loss.item(),
#                    "Train epoch": epoch})
#
#         print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
#         # use test dataset per 10 times
#         if i % 10 == 0:
#             j, data = next(enumerate(testdataloader, 0))
#             points, target = data
#             target = target[:, 0]
#             points = points.transpose(2, 1)
#             points, target = points.cuda(), target.cuda()
#             classifier = classifier.eval()
#             pred, _, _ = classifier(points)
#             loss = F.nll_loss(pred, target)
#             pred_choice = pred.data.max(1)[1]
#             correct = pred_choice.eq(target.data).cpu().sum()
#
#             wandb.log({"val acc": correct.item() / float(opt.batchSize * 2500), "val loss": loss.item(),
#                        "val epoch": epoch})
#
#             print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
#
#     torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
#
# total_correct = 0
# total_testset = 0
# for i,data in tqdm(enumerate(testdataloader, 0)):
#     points, target = data
#     target = target[:, 0]
#     points = points.transpose(2, 1)
#     points, target = points.cuda(), target.cuda()
#     classifier = classifier.eval()
#     pred, _, _ = classifier(points)
#     pred_choice = pred.data.max(1)[1]
#     correct = pred_choice.eq(target.data).cpu().sum()
#     total_correct += correct.item()
#     total_testset += points.size()[0]
#
# print("final accuracy {}".format(total_correct / float(total_testset)))