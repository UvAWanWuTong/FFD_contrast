from __future__ import print_function
import os
from utils.utils import save_config_file,save_checkpoint
from model.pointnet.model import Contrastive_PointNet, feature_transform_regularizer
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
