from __future__ import print_function
import wandb
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from model.pointnet.dataset import ModelNetDataset
from model.pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import re
import torch.backends.cudnn as cudnn
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
parser.add_argument('--outf', type=str, default='checkpoints_eval', help='output  folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument(
    '--disable_cuda', default=False,action='store_true',
                    help='Disable CUDA')
parser.add_argument(
    '--lr',type=float, default = 0.001, help='learning rate')

parser.add_argument(
    '--step_size', type=int, default=200, help='step size of learning rate decay')

parser.add_argument(
    '--decay', type=float, default=0.8, help='lr decay  ')

parser.add_argument(
    '--expriment_name', type=str, default='FFD_contrast', help='the name of current expriment ')


parser.add_argument(
    '--test',  default=False, action='store_true', help='test the run  ')

parser.add_argument(
    '--data_augmentation',  default=False, action='store_true', help='normal data  augmentation  ')


opt = parser.parse_args()
opt.expriment_name = "{lr:}_{step_size}_{decay}_cls_evaluate-{batchSize}_aug_{aug}".\
        format(lr=opt.lr, step_size=opt.step_size, decay=opt.decay,batchSize=opt.batchSize,aug=opt.data_augmentation)


if not os.path.exists(os.path.join(opt.outf, opt.expriment_name)):
    os.makedirs(os.path.join(opt.outf, opt.expriment_name))

save_path = os.path.join(opt.outf, opt.expriment_name)

if not opt.disable_cuda and torch.cuda.is_available():
    opt.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
else:
    opt.device = torch.device('cpu')
    opt.gpu_index = -1

print('Now use the device', opt.device)




blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


if opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train',
        data_augmentation=opt.data_augmentation
    )

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
dict  = torch.load(opt.model)['state_dict']
classifier.load_state_dict(torch.load(opt.model)['state_dict'],strict=True)
print('restore model successfully')

parameters_ = []
# freeze all layers but the last fc

# find last few fc layers and frozen the rest of feature extraction layers
for name, param in classifier.named_parameters():
    # if not re.match(r'^fc\d+\.(weight|bias)$',name):
    if not re.match(r'^(fc|bn)\d+\.(weight|bias)$',name):
        param.requires_grad = False
        parameters_.append(name)

parameters = list(filter(lambda p: p.requires_grad, classifier.parameters()))
assert len(parameters) == 9  # fc{1,2,3}.weight, fc{1,2,3}.bias


optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.decay)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize



max_top1 = 0
max_val_acc = 0




wandb.login(key='d27f3b3e72d749fb99315e0e86c6b36b6e23617e')
wandb.init(project="FFD_Contrast_evaluation",
           name=opt.expriment_name,
           config={
               "architecture":"pointnet-classification",
               "epochs": opt.nepoch,
               "LearningRate":opt.lr,
               "step_size":opt.step_size,
               "model":opt.model.split('/')[-3],
               "dataset":'ModelNet40'
           }
           )

print('Iinitialization of wandb complete\n')


def evaluation(testdataloader=testdataloader,model=None):
    classifier =model
    j, data = next(enumerate(testdataloader, 0))
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    val_loss = F.nll_loss(pred, target)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    val_acc = correct.item() / float(opt.batchSize)

    return val_loss,val_acc


def total_acc(testdataloader=testdataloader,model=None):
    classifier=model
    total_correct = 0
    total_testset = 0
    print('-------testing--------')
    for i, data in tqdm(enumerate(testdataloader, 0)):
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
    top1_acc = total_correct / float(total_testset)
    return top1_acc



for epoch in range(opt.nepoch):
    counter = 0

    for i, data in enumerate(dataloader, 0):
        # train
        # if counter >5:
        #     break
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
                   "Train epoch": epoch,"learning rate":scheduler.get_last_lr()[0]})
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        # counter +=1
        if i % 10 == 0:
            #evaluate
            val_loss,val_acc = evaluation(model=classifier)
            wandb.log({"val_acc": val_acc, "val loss": loss.item()})
            if val_acc> max_val_acc:
                max_val_acc = val_acc
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), val_loss.item(), correct.item()/float(opt.batchSize)))
    if epoch % 3 == 0:
        # calculate top-1 acc total
        top1_acc = total_acc(model=classifier)
        print("total accuracy {}",top1_acc)
        wandb.log({"top1_acc":top1_acc})

        # save the best model checkpoints
        if top1_acc > max_top1:
                is_best = True
                print('Save Best  model.......')
                max_top1 = top1_acc
                wandb.log({"max_top1_acc": max_top1})

        else:
                is_best = False
                print('Save check points......')

        checkpoint_name = 'checkpoint_{}.pth.tar'.format(epoch)
        save_checkpoint({
                'current_epoch': epoch,
                'epoch': opt.nepoch,
                'state_dict': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=checkpoint_name,file_dir=save_path)





