from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from model.pointnet.dataset import Contrastive_ModelNetDataset
from model.pointnet.model import Contrastive_PointNet, feature_transform_regularizer
from utils.criterion import  NCESoftmaxLoss
from utils.FFD_contrast import FFD_contrast
import torch.backends.cudnn as cudnn
import tqdm
import torch.nn.functional as F
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=3000, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument(
    '--outf', type=str, default='checkpoints', help='output folder')
parser.add_argument(
    '--model', type=str, default='', help='model path')
parser.add_argument(
    '--dataset', type=str, required=True, help="dataset path")
parser.add_argument(
    '--dataset_type', type=str, default='modelnet40', help="modelnet40")
parser.add_argument(
    '--feature_transform',default= False, action='store_true', help="use feature transform")
parser.add_argument(
    '--disable_cuda', default=False,action='store_true',
                    help='Disable CUDA')
parser.add_argument(
    '--lr',type=float, default = 0.001, help='learning rate')

parser.add_argument(
    '--ffd_points', type=int, default=27, help='number of ffd points' )
parser.add_argument(
    '--ffd_control', type=int, default=6, help='number of control points in ffd')
parser.add_argument(
    '--step_size', type=int, default=200, help='step size of learning rate decay')


def main():
    opt = parser.parse_args()
    print(opt)
    if not opt.disable_cuda and torch.cuda.is_available():
        opt.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        opt.device = torch.device('cpu')
        opt.gpu_index = -1

    print('Now use the device',opt.device )



    opt.manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset_type == 'modelnet40':
        dataset = Contrastive_ModelNetDataset(

            root=opt.dataset,
            npoints=opt.num_points,
            split='train',
            ffd_points = opt.ffd_points,
            ffd_control = opt.ffd_control

        )
    else:
        exit('wrong dataset type')

    train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
            drop_last=True,
        )


    try:
        os.makedirs(opt.outf)
    except OSError:
        pass



    model = Contrastive_PointNet(feature_transform=opt.feature_transform)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.8)


    num_batch = len(dataset) / opt.batchSize

    if opt.model != '':
        model.load_state_dict(torch.load(opt.model)['state_dict'])
        print('restore successful')
        print('current epoch:%d'% torch.load(opt.model)['current_epoch'])

    # wandb.login(key='d27f3b3e72d749fb99315e0e86c6b36b6e23617e')
    # wandb.init(project="FFD_Contrast",
    #                    name="FFD_Contrast-32",
    #                    config={
    #                        "architecture":"pointnet-classification",
    #                        "batch_size":opt.batchSize,
    #                        "epochs": opt.nepoch,
    #                        "dataset":'ModelNet40',
    #                        "ffd_points" : opt.ffd_points,
    #                        "ffd_control" : opt.ffd_control
    #                    }
    #                    )


    print('Iinitialization of logger complete\n')



    print('current batch size',opt.batchSize)
    ffd_contrast = FFD_contrast (model=model,optimizer=optimizer,scheduler=scheduler, writer=wandb, num_batch =num_batch,args =opt )
    ffd_contrast.train(train_dataloader)



if __name__ == "__main__":
    main()



