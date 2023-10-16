from __future__ import print_function
import os
from utils.utils import save_config_file,save_checkpoint
from model.pointnet.model import Contrastive_PointNet, feature_transform_regularizer,Deform_Net
from utils.criterion import  NCESoftmaxLoss
import logging
# from model.pointnet.model import Deform_Net
from tqdm.auto import tqdm
from chamferdist import ChamferDistance
from utils.emd_ import emd_module
import torch
class FFD_contrast(object):
    def __init__(self,*args,**kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = kwargs['writer']
        self.num_batch =  kwargs['num_batch']
        self.min_loss = 1000
        self.model_list =  kwargs['model_list']
        self.chamferDist = ChamferDistance()
        self.EMD = emd_module.emdModule()

    def regularization_selector(self, loss_type=None, control_points = None, point_cloud = None, classifier=None, criterion=None):

        cp1,cp2 = control_points
        pc1,pc2 = point_cloud

        if loss_type == 'none':
            return 0
        if loss_type == 'chamfer':
            return 0.5 * self.chamferDist(cp1.cpu(), cp1.cpu(), bidirectional=True).cuda() * 0.01

        if loss_type == 'emd':
            pc1 = pc1.transpose(2, 1).to(self.args.device)
            pc2 = pc2.transpose(2, 1).to(self.args.device)
            return torch.sum(self.EMD(pc1, pc2, 0.005, 300)[0]) * 0.01

        if self.args.regularization == 'double':
            # get the feature ofd the control points
            point1 = cp1.transpose(2, 1).to(self.args.device)
            point2 = cp2.transpose(2, 1).to(self.args.device)
            dp_1_feat, _, _, = classifier(point1)
            dp_2_feat, _, _, = classifier(point2)
            return criterion(dp_1_feat, dp_2_feat) * 0.01

    def train(self,train_loader):
        pass
        # for epoch in tqdm(range(self.args.nepoch)):
        #     """ contrastive learning """
        #     counter = 0
        #     epoch_loss = 0
        #     for data in  train_loader:
        #
        #         points1, points2 = data
        #         points1 = points1.transpose(2, 1).to(self.args.device)
        #         points2 = points2.transpose(2, 1).to(self.args.device)
        #
        #
        #
        #         self.optimizer.zero_grad()
        #         classifier = self.model.train()
        #         F0, trans, trans_feat = classifier(points1)
        #         F1, trans, trans_feat = classifier(points2)
        #         # if learnable FFD
        #
        #
        #         criterion = NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device)
        #         loss = criterion(F0, F1)
        #
        #         if self.args.feature_transform:
        #             loss += feature_transform_regularizer(trans_feat) * 0.001
        #
        #
        #
        #         epoch_loss  += loss.item()
        #         loss.backward()
        #         self.optimizer.step()
        #         self.scheduler.step()
        #         self.writer.log({
        #                        "train loss": loss.item(),
        #                        "Train epoch": epoch,
        #                        "Learning rate":self.scheduler.get_last_lr()[0]})
        #
        #         print('\n [%d: %d/%d]  loss: %f  lr: %f' % ( epoch, counter, self.num_batch, loss.item(),self.scheduler.get_last_lr()[0]))
        #         counter +=1
        #     if epoch % 1 ==0:
        #         # save the best model checkpoints
        #         if epoch_loss / self.num_batch < self.min_loss:
        #                 is_best = True
        #                 print('Save Best model')
        #         else:
        #                 is_best = False
        #                 print('Save check points......')
        #
        #         checkpoint_name = 'check_point{}.pth.tar'.format(epoch)
        #         save_checkpoint({
        #             'current_epoch': epoch,
        #             'epoch': self.args.nepoch,
        #             'state_dict': self.model.state_dict(),
        #             'optimizer': self.optimizer.state_dict(),
        #         }, is_best=is_best, filename=checkpoint_name,file_dir=self.args.save_path)
        #         self.min_loss = loss

