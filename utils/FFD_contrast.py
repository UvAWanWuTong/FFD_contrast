from __future__ import print_function
import os
from utils.utils import save_config_file,save_checkpoint
from model.pointnet.model import Contrastive_PointNet, feature_transform_regularizer
from utils.criterion import  NCESoftmaxLoss
import logging
from tqdm.auto import tqdm



class FFD_contrast(object):
    def __init__(self,*args,**kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = kwargs['writer']
        self.num_batch =  kwargs['num_batch']

    def train(self,train_loader):
        for epoch in tqdm(range(self.args.nepoch)):
            """ contrastive learning """
            counter = 0

            for data in  train_loader:
                points1, points2 = data
                points1 = points1.transpose(2, 1).cuda()
                points2 = points2.transpose(2, 1).cuda()
                self.optimizer.zero_grad()
                classifier = self.model.train()
                F0, trans, trans_feat = classifier(points1)
                F1, trans, trans_feat = classifier(points2)
                criterion = NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device)
                loss = criterion(F0, F1)

                if self.args.feature_transform:
                    loss += feature_transform_regularizer(trans_feat) * 0.001

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.writer.log({"train loss": loss.item(),
                           "Train epoch": epoch,
                           "Learning rate":self.scheduler.get_last_lr()[0]})
                print('\n [%d: %d/%d]  loss: %f  lr: %f' % ( epoch, counter, self.num_batch, loss.item(),self.scheduler.get_last_lr()[0]))
                counter +=1

        logging.info("Training has finished.")
            # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.nepoch)
        save_checkpoint({
                'epoch': self.args.nepoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=True, filename=os.path.join(self.args.outf, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.args.outf}.")






