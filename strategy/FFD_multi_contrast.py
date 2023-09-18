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
import numpy as np
from torch  import nn

from utils.emd_ import emd_module
# from utils.cd.chamferdist import ChamferDistance as CD
from chamferdist import ChamferDistance


class FFD_multi_contrast(object):
    def __init__(self,*args,**kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = kwargs['writer']
        self.num_batch =  kwargs['num_batch']
        self.min_loss = 1000
        self.model_list =  kwargs['model_list']
        self.mixrates= 0.5
        self.alpha = 0.5
        self.EMD = emd_module.emdModule()
        self.chamferDist = ChamferDistance()


    def pointmixup(self,align,mixrates,xyz1,xyz2):
        # mix_rate = torch.tensor(mixrates).to(self.args.device).float()
        # mix_rate = mix_rate.unsqueeze_(1).unsqueeze_(2)
        # mix_rate_expand_xyz = mix_rate.expand(xyz1.shape).to(self.args.device)
        _, ass = self.EMD(xyz1, xyz2, 0.005, 300)
        B = xyz1.shape[0]
        ass=ass.cpu().numpy()
        for i in range(B):
            xyz2[i] = xyz2[i][ass[i]]
        # xyz = xyz1 * (1 - mix_rate_expand_xyz) + xyz2 * mix_rate_expand_xyz
        xyz = xyz1 * (1 -mixrates) + xyz2 * mixrates


        return xyz

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
                classifier = self.model_list[0].train()
                deform_net_1 = self.model_list[1].train()
                deform_net_2 = self.model_list[2].train()



                feature, _, _ = classifier(points)


                n_feature = normalize_pointcloud_tensor(feature)

                # get FFD deformation strategy
                # FFD learnable
                dp_1 = deform_net_1(n_feature).to(self.args.device)
                dp_2 = deform_net_2(n_feature).to(self.args.device)



                # perfom ffd
                points1_ffd = torch.bmm(b,p+dp_1)
                points2_ffd = torch.bmm(b,p+dp_2)

                # normalization
                points1_ffd = normalize_pointcloud_tensor(points1_ffd)
                points2_ffd = normalize_pointcloud_tensor(points2_ffd)

                if self.args.regularization:


                    cd0, cd1, _, _ = self.cd(points1_ffd, points1_ffd)
                    loss_chamfer = self.chamferDist(points1_ffd, points2_ffd, bidirectional=True)

                # B = points2_ffd.shape[0]
                # mixrates = (0.5 - np.abs(np.random.beta(0.5, 0.5, B) - 0.5))
                mixrates = 0.5
                points3 = self.pointmixup(False,mixrates,points1_ffd,points2_ffd)
                points3 = normalize_pointcloud_tensor(points3)






                # calculate the chamfer distances
                # dist = self.chamferDist(points1_ffd, points2_ffd)
                # dist = dist.detach().cpu().item()

                points1_ffd = points1_ffd.transpose(2, 1).to(self.args.device)
                points2_ffd = points2_ffd.transpose(2, 1).to(self.args.device)
                points3 = points3.transpose(2, 1).to(self.args.device)
                # get the feature after FFD
                F1, _, _, = classifier(points1_ffd)
                F2, _, _, = classifier(points2_ffd)
                F3, _, _, = classifier(points3)

                # get the feature ofd the control points

                dp_1 = dp_1.transpose(2, 1).to(self.args.device)
                dp_2 = dp_2.transpose(2, 1).to(self.args.device)

                criterion = NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device)

                # NCE loss after deformed objects

                if self.args.regularization:

                    loss = 0.1 * criterion(F1, F3) + 0.1 * criterion(F2, F3) + 0.8 * criterion(F1, F2) - loss_chamfer
                else:
                    loss = 0.1 * criterion(F1, F3) + 0.1 * criterion(F2, F3) + 0.8 * criterion(F1, F2)










                # NCE loss afte deformed control points







                epoch_loss  += loss.item()

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()



                if self.args.regularization:
                    self.writer.log({
                                   "train loss": loss.item(),
                                   "chamfer loss":loss_chamfer.item(),
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
                deform_net_name = 'deform_net_1.pth.tar'
                save_checkpoint({
                    'state_dict': deform_net_1.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=deform_net_name, file_dir=self.args.save_path,save_deform=True)
                self.min_loss = loss


                deform_net_name = 'deform_net_2.pth.tar'
                save_checkpoint({
                    'state_dict': deform_net_2.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=deform_net_name, file_dir=self.args.save_path,save_deform=True)
                self.min_loss = loss






