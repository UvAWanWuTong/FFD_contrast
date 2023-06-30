from __future__ import print_function
import os
from utils.utils import save_config_file,save_checkpoint
from model.pointnet.model import Contrastive_PointNet, feature_transform_regularizer,Deform_Net
from utils.criterion import  NCESoftmaxLoss
import logging
from tqdm.auto import tqdm

need_pytorch3d=False
import sys
import torch

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)



class FFD_learnable_contrast(object):
    def __init__(self,*args,**kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = kwargs['writer']
        self.num_batch =  kwargs['num_batch']
        self.min_loss = 1000
        self.model_list =  kwargs['model_list']







    def train(self,train_loader):
        for epoch in tqdm(range(self.args.nepoch)):
            """ contrastive learning """
            counter = 0
            epoch_loss = 0
            for data in  train_loader:
                (b1,p1), (b2,p2) = data
                b1 = b1.to(self.args.device)
                p1 = p1.to(self.args.device)
                b2 = b2.to(self.args.device)
                p2 = p2.to(self.args.device)




                # perfom FFD deformation

                points1 = torch.bmm(b1, p1)
                points2 = torch.bmm(b2, p2)

                #points normlaize

                points1 = points1.transpose(2, 1).to(self.args.device)
                points2 = points2.transpose(2, 1).to(self.args.device)

                self.optimizer.zero_grad()
                classifier = self.model_list[0].train()
                deform_net_1 = self.model_list[1].train()
                deform_net_2 = self.model_list[2].train()



                # 先变形 在产生 Feature


                F1, trans, trans_feat = classifier(points1)
                F2, trans, trans_feat = classifier(points2)


                # get FFD deformation strategy

                # FFD learnable
                dp_1 = deform_net_1(F1).to(self.args.device)
                dp_2 = deform_net_2(F2).to(self.args.device)


                # perfom ffd
                points1_ffd = torch.bmm(b1,p1+dp_1)
                points2_ffd = torch.bmm(b1,p2+dp_2)

                loss_chamfer, _ = chamfer_distance(points1_ffd, points2_ffd)

                dist = loss_chamfer.detach().cpu().numpy()

                print(dist)

                points1_ffd = points1_ffd.transpose(2, 1).to(self.args.device)
                points2_ffd = points2_ffd.transpose(2, 1).to(self.args.device)

                # get the feature after FFD
                F1, trans, trans_feat, = classifier(points1_ffd)
                F2, trans, trans_feat, = classifier(points2_ffd)




                criterion = NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device)
                loss = criterion(F1, F2)

                if self.args.feature_transform:
                    loss += feature_transform_regularizer(trans_feat) * 0.001
                epoch_loss  += loss.item()
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                self.writer.log({
                               "train loss": loss.item(),
                               "Train epoch": epoch,
                               "Learning rate":self.scheduler.get_last_lr()[0],
                               # "chamferDist":dist,
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
                    'state_dict': deform_net_1.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=deform_net_name, file_dir=self.args.save_path,save_deform=True)
                self.min_loss = loss








