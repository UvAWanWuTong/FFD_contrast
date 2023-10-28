from __future__ import print_function
import os
from utils.utils import save_config_file,save_checkpoint
from utils.criterion import  NCESoftmaxLoss
import logging
from tqdm.auto import tqdm

import sys
import torch
import os
from utils.utils import save_config_file,save_checkpoint,normalize_pointcloud_tensor
from utils.criterion import  NCESoftmaxLoss
import logging
from tqdm.auto import tqdm
import sys
import torch
from torch  import nn
# from chamferdist import ChamferDistance
from utils.emd_ import emd_module

from strategy.FFD_contrast import FFD_contrast

class FFD_random_contrast(FFD_contrast):
    def __init__(self,*args,**kwargs):
        super(FFD_random_contrast, self).__init__(*args,**kwargs)



    def train(self,train_loader):
        for epoch in tqdm(range(self.args.nepoch)):
            """ contrastive learning """
            counter = 0
            epoch_loss = 0
            for data in  train_loader:
                (b,p) = data
                b = b.to(self.args.device)
                p = p.to(self.args.device)





                self.optimizer.zero_grad()
                classifier = self.model.train()

                # create a uniform distribution
                random_sampler = torch.distributions.uniform.Uniform(p.min(), p.max())

                # sample  random numbers from the distribution as FFD parameters
                dp_1 = random_sampler.sample(sample_shape=( p.shape[0],p.shape[1],p.shape[2]))
                dp_2 = random_sampler.sample(sample_shape=( p.shape[0],p.shape[1],p.shape[2]))

                # random FFD

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


                criterion = NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device)

                # NCE loss after deformed objects

                reg_loss = self.regularization_selector(loss_type=self.args.regularization,point1=(p+dp_1),point2=(p+dp_2),classifier=classifier,criterion=criterion)

                loss = criterion(F1, F2) - reg_loss

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



