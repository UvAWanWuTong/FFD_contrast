from __future__ import print_function
import os
from utils.utils import save_config_file,save_checkpoint
from model.pointnet.model import Contrastive_PointNet, feature_transform_regularizer,Deform_Net
from utils.criterion import  NCESoftmaxLoss
import logging
from tqdm.auto import tqdm

import sys
import torch
import os
from utils.utils import save_config_file,save_checkpoint,normalize_pointcloud_tensor
from model.pointnet.model import Contrastive_PointNet, feature_transform_regularizer,Deform_Net
from utils.criterion import  NCESoftmaxLoss
import logging
from tqdm.auto import tqdm
import sys
import torch
from torch  import nn


class FFD_mix_contrast(object):
    def __init__(self,*args,**kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = kwargs['writer']
        self.num_batch =  kwargs['num_batch']
        self.min_loss = 1000
        self.model_list =  kwargs['model_list']
        # self.regularization =  kwargs['regularization']
        # self.chamferDist = ChamferDistance()


    def train(self,train_loader):
        for epoch in tqdm(range(self.args.nepoch)):
            """ contrastive learning """
            counter = 0
            epoch_loss = 0
            for data in  train_loader:
                (b,p) = data
                b = b.to(self.args.device)
                p = p.to(self.args.device)





                # perfom FFD deformation

                points = torch.bmm(b, p)

                #points normlaize

                points = points.transpose(2, 1).to(self.args.device)

                self.optimizer.zero_grad()

                # mix FFD strategy

                classifier = self.model_list[0].train()
                deform_net = self.model_list[1].train()

                random_sampler = torch.distributions.uniform.Uniform(p.min(), p.max())





                feature, _, _ = classifier(points)


                n_feature = normalize_pointcloud_tensor(feature)

                # get FFD deformation strategy               # FFD learnable
                dp_1 = deform_net(n_feature).to(self.args.device)
                dp_2 = random_sampler.sample(sample_shape=( p.shape[0],p.shape[1],p.shape[2]))





                # perfom ffd
                points1_ffd = torch.bmm(b,p+dp_1)
                points2_ffd = torch.bmm(b,p+dp_2)

                # normalization
                points1_ffd = normalize_pointcloud_tensor(points1_ffd)
                points2_ffd = normalize_pointcloud_tensor(points2_ffd)

                # calculate the chamfer distances
                # dist = self.chamferDist(points1_ffd, points2_ffd)
                # dist = dist.detach().cpu().item()

                points1_ffd = points1_ffd.transpose(2, 1).to(self.args.device)
                points2_ffd = points2_ffd.transpose(2, 1).to(self.args.device)
                # get the feature after FFD
                F1, _, _, = classifier(points1_ffd)
                F2, _, _, = classifier(points2_ffd)

                # get the feature ofd the control points

                dp_1 = dp_1.transpose(2, 1).to(self.args.device)
                dp_2 = dp_2.transpose(2, 1).to(self.args.device)

                criterion = NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device)

                # NCE loss after deformed objects
                loss = criterion(F1, F2)

                # NCE loss afte deformed control points





                if self.args.regularization:
                    dp_1_feat, _, _, = classifier(dp_1)
                    dp_2_feat, _, _, = classifier(dp_2)
                    loss_dp = criterion(dp_1_feat, dp_2_feat) * 0.01
                    loss -= loss_dp

                epoch_loss  += loss.item()

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()


                if self.args.regularization:
                    self.writer.log({
                                   "train loss": loss.item(),
                                   "dp loss":loss_dp.item(),
                                   "Train epoch": epoch,
                                   "Learning rate":self.scheduler.get_last_lr()[0],



                                   },
                                  )
                else:
                    self.writer.log({
                        "train loss": loss.item(),
                        "Train epoch": epoch,
                        "Learning rate": self.scheduler.get_last_lr()[0],

                    },
                    )



                print('\n [%d: %d/%d]  loss: %f  lr: %f' % ( epoch, counter, self.num_batch, loss.item(),self.scheduler.get_last_lr()[0]))
                # counter +=1
                # if counter > 5:
                #     break
            if epoch % 5 ==0:
                # save the best model checkpoints
                if epoch_loss / self.num_batch < self.min_loss:
                        is_best = True
                        print('Save Best model')
                else:
                        is_best = False
                        print('Save check points......')

                checkpoint_name = 'check_point{}.pth.tar'.format(epoch)
                save_checkpoint({
                    'current_epoch': epoch,
                    'epoch': self.args.nepoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=checkpoint_name,file_dir=self.args.save_path)
                self.min_loss = loss


                #save deform net
                deform_net_name = 'deform_net_mix.pth.tar'
                save_checkpoint({
                    'state_dict': deform_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=deform_net_name, file_dir=self.args.save_path,save_deform=True)
                self.min_loss = loss




