from __future__ import print_function


import sys
import torch
import os
from utils.utils import save_config_file,save_checkpoint,normalize_pointcloud_tensor
from utils.criterion import  NCESoftmaxLoss
import logging
from tqdm.auto import tqdm
import sys
import torch
import numpy as np
from torch  import nn

from utils.emd_ import emd_module
from chamferdist import ChamferDistance
from torch.utils.data import DataLoader
from Data.dataset import  ModelNet40SVM
from sklearn.svm import SVC
import numpy as np

torch.autograd.set_detect_anomaly(True)

from strategy.FFD_contrast import FFD_contrast

class FFD_multi_contrast(FFD_contrast):
    def __init__(self,*args,**kwargs):
        super(FFD_multi_contrast, self).__init__(*args,**kwargs)
        self.classifier = self.model_list[0].train()
        self.deform_net_1 = self.model_list[1].train()
        self.deform_net_2 = self.model_list[2].train()
        self.writer.watch(self.classifier)
        self.test_freq = 1000
        self.sigmoid = nn.Sigmoid()



    def pointmixup(self,mixrates,xyz1,xyz2):
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


                dp_1  = self.sigmoid(dp_1)
                dp_2  = self.sigmoid(dp_2)

                # perfom ffd
                points1_ffd = torch.bmm(b,p+dp_1)
                points2_ffd = torch.bmm(b,p+dp_2)

                # normalization
                points1_ffd = normalize_pointcloud_tensor(points1_ffd)
                points2_ffd = normalize_pointcloud_tensor(points2_ffd)






                # B = points2_ffd.shape[0]
                # mixrates = (0.5 - np.abs(np.random.beta(0.5, 0.5, B) - 0.5))
                mixrates = 0.5
                points3 = self.pointmixup(mixrates,points1_ffd,points2_ffd)
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



                criterion = NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device)

                # NCE loss after deformed objects

                term_1 = criterion(F1, F3)
                term_2 = criterion(F2, F3)
                term_3 = criterion(F1, F2)

                reg_loss = self.regularization_selector(loss_type=self.args.regularization,control_points=((p+dp_1),(p+dp_2)),point_cloud=(points1_ffd,points2_ffd),classifier=classifier,criterion=criterion)
                loss = 0.1 * term_1 + 0.1 * term_2 + 0.8 * term_3 - reg_loss










                # NCE loss afte deformed control points







                epoch_loss  += loss.item()

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()



                if self.args.regularization:
                    self.writer.log({
                                   "train loss": loss.item(),
                                   "chamfer loss":reg_loss.item(),
                                   "Train epoch": epoch,
                                   "Learning rate":self.scheduler.get_last_lr()[0],
                                   "loss 1-3":term_1.item(),
                                   "loss 2-3": term_2.item(),
                                   "loss 1-2": term_3.item(),

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

    def train_PointNet(self, train_loader):
        for epoch in tqdm(range(self.args.nepoch)):
            """ contrastive learning """
            counter = 0
            epoch_loss = 0
            for data in train_loader:
                (b, p) = data
                b = b.to(self.args.device)
                p = p.to(self.args.device)

                points = torch.bmm(b, p)
                points = points.transpose(2, 1).to(self.args.device)
                self.optimizer.zero_grad()

                projection_feature, _, _ = self.classifier(points)
                n_feature = normalize_pointcloud_tensor(projection_feature)

                # get FFD deformation strategy
                # FFD learnable
                dp_1 = self.deform_net_1(n_feature).to(self.args.device)
                dp_2 = self.deform_net_2(n_feature).to(self.args.device)

                dp_1 = self.sigmoid(dp_1)
                dp_2 = self.sigmoid(dp_2)

                # perfom ffd
                points1_ffd = torch.bmm(b, p + dp_1)
                points2_ffd = torch.bmm(b, p + dp_2)

                # normalization
                points1_ffd = normalize_pointcloud_tensor(points1_ffd)
                points2_ffd = normalize_pointcloud_tensor(points2_ffd)

                # B = points2_ffd.shape[0]
                # mixrates = (0.5 - np.abs(np.random.beta(0.5, 0.5, B) - 0.5))
                mixrates = 0.5
                points3 = self.pointmixup(mixrates,points1_ffd,points2_ffd)
                points3 = normalize_pointcloud_tensor(points3)



                # calculate the chamfer distances
                # dist = self.chamferDist(points1_ffd, points2_ffd)
                # dist = dist.detach().cpu().item()

                points1_ffd = points1_ffd.transpose(2, 1).to(self.args.device)
                points2_ffd = points2_ffd.transpose(2, 1).to(self.args.device)
                points3 = points3.transpose(2, 1).to(self.args.device)

                # get the feature after FFD
                F1, _, _, = self.classifier(points1_ffd)
                F2, _, _, = self.classifier(points2_ffd)
                F3, _, _, = self.classifier(points3)



                criterion = NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device)

                # NCE loss after deformed objects

                term_1 = criterion(F1, F3)
                term_2 = criterion(F2, F3)
                term_3 = criterion(F1, F2)



                if self.args.regularization != 'none':
                    reg_loss = self.regularization_selector(loss_type=self.args.regularization,
                                                            control_points=((p + dp_1), (p + dp_2)),
                                                            point_cloud=(points1_ffd, points2_ffd),
                                                            classifier=self.classifier,
                                                            criterion=NCESoftmaxLoss(batch_size=self.args.batchSize,
                                                                                     cur_device=self.args.device))
                    loss =   term_1 +  term_2 + term_3

                else:
                    loss =  term_1 +  term_2 + term_3
                # Testing

                epoch_loss += loss.item()
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                if self.args.regularization != 'none':
                    self.writer.log({
                        "train loss": loss.item(),
                        "reg loss": reg_loss.item(),
                        "Train epoch": epoch,
                        "Learning rate": self.scheduler.get_last_lr()[0],
                        "Max ACC": self.best_acc

                    },
                    )
                else:
                    self.writer.log({
                        "train loss": loss.item(),
                        "Train epoch": epoch,
                        "Learning rate": self.scheduler.get_last_lr()[0],
                        "Max ACC": self.best_acc

                    },
                    )

                print('\n [%d: %d/%d]  loss: %f  lr: %f' % (
                    epoch, counter, self.num_batch, loss.item(), self.scheduler.get_last_lr()[0]))

            train_val_loader = DataLoader(ModelNet40SVM(partition='train', root=self.args.dataset),
                                          batch_size=self.args.batchSize,
                                          shuffle=True)
            test_val_loader = DataLoader(ModelNet40SVM(partition='test', root=self.args.dataset),
                                         batch_size=self.args.batchSize,
                                         shuffle=True)
            # Testing on modelnet40 data
            feats_train = []
            labels_train = []
            self.classifier.eval()
            for i, (data, label) in enumerate(train_val_loader):
                labels = list(map(lambda x: x[0], label.numpy().tolist()))
                data = data.permute(0, 2, 1).to(self.args.device)
                with torch.no_grad():
                    feats = self.classifier(data)[1]
                feats = feats.detach().cpu().numpy()
                for feat in feats:
                    feats_train.append(feat)
                labels_train += labels

            feats_train = np.array(feats_train)
            labels_train = np.array(labels_train)

            feats_test = []
            labels_test = []

            for i, (data, label) in enumerate(test_val_loader):
                labels = list(map(lambda x: x[0], label.numpy().tolist()))
                data = data.permute(0, 2, 1).to(self.args.device)
                with torch.no_grad():
                    feats = self.classifier(data)[1]
                feats = feats.detach().cpu().numpy()
                for feat in feats:
                    feats_test.append(feat)
                labels_test += labels

            feats_test = np.array(feats_test)
            labels_test = np.array(labels_test)

            model_tl = SVC(C=0.1, kernel='linear')
            model_tl.fit(feats_train, labels_train)
            test_accuracy = model_tl.score(feats_test, labels_test)
            print(f"Linear Accuracy : {test_accuracy}")
            self.writer.log({"Linear Accuracy": test_accuracy})

            if epoch % 5 == 0:
                # save the best model checkpoints
                if test_accuracy > self.best_acc:
                    is_best = True
                    print('Save Best model')
                    self.best_acc = test_accuracy
                else:
                    is_best = False
                    print('Save check points......')

                checkpoint_name = 'check_point{}.pth.tar'.format(epoch)
                save_checkpoint({
                    'current_epoch': epoch,
                    'epoch': self.args.nepoch,
                    'state_dict': self.classifier.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=checkpoint_name, file_dir=self.args.save_path)

                # save deform net
                deform_net_name = 'deform_net_1.pth.tar'
                save_checkpoint({
                    'state_dict': self.deform_net_1.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=deform_net_name, file_dir=self.args.save_path, save_deform=True)

                deform_net_name = 'deform_net_2.pth.tar'
                save_checkpoint({
                    'state_dict': self.deform_net_2.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=deform_net_name, file_dir=self.args.save_path, save_deform=True)








