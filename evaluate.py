from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from model.pointnet.dataset import ModelNetDataset
from model.pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import re
import torch.backends.cudnn as cudnn
from utils.sampler import Normalize
from utils.utils import  save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls_eval', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument(
    '--disable_cuda', default=False,action='store_true',
                    help='Disable CUDA')

opt = parser.parse_args()
print(opt)
if not opt.disable_cuda and torch.cuda.is_available():
    opt.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
else:
    opt.device = torch.device('cpu')
    opt.gpu_index = -1

print('Now use the device', opt.device)



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
        split='train')

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
    num_workers=int(opt.workers))

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


assert opt.model != '', "The model parameter should not be empty"

classifier.load_state_dict(torch.load(opt.model)['state_dict'],strict=False)
print('restore model successfully')

parameters_ = []
# freeze all layers but the last fc

# find last few fc layers and frozen the rest of feature extraction layers
for name, param in classifier.named_parameters():
    if not re.match(r'^fc\d+\.(weight|bias)$',name):
                param.requires_grad = False
                parameters_.append(name)

parameters = list(filter(lambda p: p.requires_grad, classifier.parameters()))
assert len(parameters) == 6  # fc{1,2,3}.weight, fc{1,2,3}.bias


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize



min_loss = 1000



wandb.login(key='d27f3b3e72d749fb99315e0e86c6b36b6e23617e')
wandb.init(project="FDD_Contrast-evaluation",
           name="pointnet",
           config={
               "architecture":"pointnet-classification",
               "epochs": opt.nepoch,
               "dataset":'ModelNet40'
           }
           )

print('Iinitialization of wandb complete\n')



for epoch in range(opt.nepoch):

    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0].to()
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        scheduler.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        wandb.log({"train acc": correct.item() / float(opt.batchSize), "train loss": loss.item(),
                   "Train epoch": epoch})
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        if i % 10 == 0:
            val_loss = 0
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            wandb.log({"val acc": correct.item() / float(opt.batchSize), "val loss": loss.item()})

            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
            if i % 100 == 0 and val_loss < min_loss:
                # save the best model checkpoints
                print('Save best model......')
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(opt.nepoch)
                save_checkpoint({
                    'current_epoch': epoch,
                    'epoch': opt.nepoch,
                    'state_dict': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=True, filename=os.path.join(opt.outf, checkpoint_name))
                min_loss = val_loss
    # torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))


total_correct = 0
total_testset = 0

for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
wandb.log({"final accuracy": total_correct / float(total_testset)})
