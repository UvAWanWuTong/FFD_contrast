from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from model.pointnet.dataset import Contrastive_ModelNetDataset
from model.pointnet.model import Contrastive_PointNet, feature_transform_regularizer
from utils.criterion import  NCESoftmaxLoss
import tqdm


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
    dataset = Contrastive_ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train')

    test_dataset = Contrastive_ModelNetDataset(
        root=opt.dataset,
        split='val',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')



dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    drop_last=True,
)

Test_dataloader= torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    drop_last=True,
)








try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = Contrastive_PointNet(feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize


# wandb.login(key='d27f3b3e72d749fb99315e0e86c6b36b6e23617e')
#
# wandb.init(project="FFD_Contrast",
#            name="FFD_Contrast-classification-32",
#            config={
#                "architecture":"pointnet-classification",
#                "epochs": opt.nepoch,
#                "dataset":'ModelNet40'
#            }
#            )
#
# print('Iinitialization of wandb complete\n')
# print('current batch size',opt.batchSize)

cur_device = torch.cuda.current_device()
min_loss = 1000000
for epoch in range(opt.nepoch):
    """ contrastive learning """

    for i, data in tqdm(enumerate(dataloader, 0)):

        pcd0,pcd1 = data
        points1, points2 = pcd0.cuda(), pcd1.cuda()
        points1 = points1.transpose(2, 1)
        points2 = points2.transpose(2, 1)
        optimizer.zero_grad()
        classifier = classifier.train()
        F0,trans,trans_feat = classifier(points1)
        F1,trans,trans_feat = classifier(points2)
        criterion = NCESoftmaxLoss(batch_size=opt.batchSize,cur_device=cur_device).cuda()
        loss = criterion(F0, F1)

        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        loss.backward()
        optimizer.step()
        # wandb.log({"train loss": loss.item(),
        #            "Train epoch": epoch})
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

    if i % 10 == 0:
            pcd0, pcd1  = data
            points1, points2 = pcd0.cuda(), pcd1.cuda()
            points1 = points1.transpose(2, 1)
            points2 = points2.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            F0, trans, trans_feat = classifier(points1)
            F1, trans, trans_feat = classifier(points2)
            criterion = NCESoftmaxLoss(batch_size=opt.batchSize, cur_device=cur_device).cuda()
            val_loss = criterion(F0, F1)
            print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, val_loss.item()))

            if val_loss < min_loss:
                min_loss = val_loss
                print("save model")
                torch.save(classifier.state_dict(), '%s/best_model.pth' % (opt.outf))

