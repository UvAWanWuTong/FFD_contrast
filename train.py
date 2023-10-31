import wandb
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from model.pointnet.dataset import Contrastive_ModelNetDataset
from model.pointnet.model import Contrastive_PointNet, feature_transform_regularizer,Deform_Net_1layer,Deform_Net_2layer,Deform_Net_3layer
from utils.criterion import  NCESoftmaxLoss
from strategy.FFD_contrast import FFD_contrast
from strategy.FFD_learnable_contrast import FFD_learnable_contrast
from strategy.FFD_random_contrast import FFD_random_contrast
from strategy.FFD_mix_contrast import FFD_mix_contrast
from strategy.FFD_multi_contrast import FFD_multi_contrast
import torch.backends.cudnn as cudnn



parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=3072, help='input point size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument(
    '--outf', type=str, default='checkpoints_train', help='output folder')
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
    '--ffd_points_axis', type=int, default=3, help='number of ffd points on each axis' )
parser.add_argument(
    '--ffd_control', type=int, default=6, help='number of control points in ffd')
parser.add_argument(
    '--step_size', type=int, default=200, help='step size of learning rate decay')
parser.add_argument(
    '--decay', type=float, default=0.8, help='lr decay  ')
parser.add_argument(
    '--sampler', type=str, default='random',help='choose of sampler'
)
parser.add_argument(
    '--task_type', type=str, default='learnable',help='type of ffd deformation, avaliable choices: random,mix,multi'
)
parser.add_argument(
    '--regularization', type=str ,default='double', help='use of regulariztion term, avaliable choices: double,chamfer,emd,none'
)

parser.add_argument(
    "--structure", type=str,default="1layer" ,choices=["1layer", "2layer", "3layer"],help='Choose of the structure of DeformNet'
)

parser.add_argument(
    '--feature_size', type=int, default=128, help='the feature size of encoder output'
)

parser.add_argument(
    '--non_linear', default=False,action='store_true',  help='non linear projection head'
)





def main():



    opt = parser.parse_args() 
    opt.ffd_points = pow(opt.ffd_points_axis,3)
    if opt.regularization != 'none':
        opt.expriment_name = "{lr:}_{step_size}_{decay}_FFD_Contrast_{task_type}_{ffd_points}_train-{batchSize}_{structure}_{feature_size}_non_linear:{non_linear}_{reg}".\
            format(lr=opt.lr, step_size=opt.step_size, decay=opt.decay,task_type=opt.task_type, ffd_points=opt.ffd_points, batchSize=opt.batchSize,structure=opt.structure,reg=opt.regularization,feature_size=opt.feature_size,non_linear=opt.non_non_linear)
    else:
        opt.expriment_name = "{lr:}_{step_size}_{decay}_FFD_Contrast_{task_type}_{ffd_points}_train-{batchSize}_{structure}_{feature_size}_non_linear:{non_linear}".\
            format(lr=opt.lr, step_size=opt.step_size, decay=opt.decay,task_type=opt.task_type, ffd_points=opt.ffd_points, batchSize=opt.batchSize,structure=opt.structure,feature_size=opt.feature_size,non_linear=opt.non_non_linear)


    if not os.path.exists(os.path.join(opt.outf,opt.expriment_name)):
        os.makedirs(os.path.join(opt.outf,opt.expriment_name))
    opt.save_path = os.path.join(opt.outf,opt.expriment_name)

    print(opt.expriment_name)
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
    # torch.manual_seed(opt.manualSeed)

    if opt.dataset_type == 'modelnet40':

            # random FFD and customized FFD
            dataset = Contrastive_ModelNetDataset(
                    root=opt.dataset,
                    npoints=opt.num_points,
                    split='train',
                    ffd_points_axis = opt.ffd_points_axis,
                    ffd_control = opt.ffd_control,
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


    model_list = None
    model = Contrastive_PointNet(feature_transform=opt.feature_transform)

    deform_net_map = {
        "1layer": Deform_Net_1layer,
        "2layer": Deform_Net_2layer,
        "3layer": Deform_Net_3layer
    }


    # try:
    #     opt.task_type not in ['leanable','random']
    # except Exception :
    #      print('No avaliable task type ')

    if opt.task_type != 'random':

        deform_net1 =  deform_net_map[opt.structure](in_features=128,out_features=(opt.ffd_points_axis+1)**3 * 3).to(opt.device)
        deform_net2 =  deform_net_map[opt.structure](in_features=128,out_features=(opt.ffd_points_axis+1)**3 * 3).to(opt.device)

        optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': deform_net1.parameters(), 'lr': 1e-3},
            {'params': deform_net2.parameters(), 'lr': 1e-3},
        ],lr=opt.lr, betas=(0.9, 0.999))

        model_list = [model,deform_net1,deform_net2]

    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))





    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.decay)




    num_batch = len(dataset) / opt.batchSize

    if opt.model != '':
        model.load_state_dict(torch.load(opt.model)['state_dict'])
        print('restore successful')
        print('current epoch:%d'% torch.load(opt.model)['current_epoch'])

    wandb.login(key='d27f3b3e72d749fb99315e0e86c6b36b6e23617e')
    wandb.init(project="FFD_Contrast_{task_type}".format(task_type = opt.task_type),
                       name=opt.expriment_name,
                       config={
                           "architecture":"pointnet-classification",
                           "batch_size":opt.batchSize,
                           "epochs": opt.nepoch,
                           "dataset":'ModelNet40',
                           "ffd_points" : opt.ffd_points,
                           "ffd_control" : opt.ffd_control,
                           "lr" : opt.lr,
                           "step_size" : opt.step_size,
                           "decay" : opt.decay,
                           "deformnet_architecture": opt.structure,
                           "task_type":opt.task_type,
                       }
                       )


    print('Iinitialization of logger complete\n')



    print('current batch size',opt.batchSize)


    if opt.task_type == "learnable":
         ffd_contrast = FFD_learnable_contrast (model=model,optimizer=optimizer,scheduler=scheduler, writer=wandb, num_batch =num_batch, args =opt, model_list=model_list)

    elif opt.task_type == "random":
        ffd_contrast = FFD_random_contrast(model=model, optimizer=optimizer, scheduler=scheduler, writer=wandb, num_batch=num_batch, args=opt)

    elif opt.task_type == "mix":
        ffd_contrast = FFD_mix_contrast(model=model, optimizer=optimizer, scheduler=scheduler, writer=wandb, num_batch=num_batch, args=opt, model_list=model_list)

    elif opt.task_type == "multi":
        ffd_contrast = FFD_multi_contrast(model=model, optimizer=optimizer, scheduler=scheduler, writer=wandb,
                                              num_batch=num_batch, args=opt, model_list=model_list)

    ffd_contrast.train(train_dataloader)



if __name__ == "__main__":
    main()



