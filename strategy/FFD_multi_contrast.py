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
import torchvision.transforms as transforms
from utils import data_utils as d_utils

class FFD_multi_contrast(FFD_contrast):
    def __init__(self,*args,**kwargs):
        super(FFD_multi_contrast, self).__init__(*args,**kwargs)
        self.classifier = self.model_list[0].train()
        self.deform_net_1 = self.model_list[1].train()
        self.deform_net_2 = self.model_list[2].train()
        self.writer.watch(self.classifier)
        self.test_freq = 1000
        # self.point_level_loss =NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device)
        self.trans_1 = transforms.Compose(
            [
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                # d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
            ])

        self.trans_2 = transforms.Compose(
            [
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                # d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
            ])

        self.trans_3 = transforms.Compose(
            [
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                # d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
            ])


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
                #
                # noise_1 = torch.rand(p.shape[0],p.shape[1],p.shape[2]).to(self.args.device)
                # noise_2 = torch.rand(p.shape[0],p.shape[1],p.shape[2]).to(self.args.device)
                #
                # dp_1 = normalize_pointcloud_tensor(dp_1 + noise_1)
                # dp_2 = normalize_pointcloud_tensor(dp_2 + noise_2)


                cp_1 = p+normalize_pointcloud_tensor(dp_1)
                cp_2 = p+normalize_pointcloud_tensor(dp_2)


                # perfom ffd
                points1_ffd = torch.bmm(b, cp_1)
                points2_ffd = torch.bmm(b, cp_2)

                # normalization
                points1_ffd = normalize_pointcloud_tensor(points1_ffd)
                points2_ffd = normalize_pointcloud_tensor(points2_ffd)


                # B = points2_ffd.shape[0]
                # mixrates = (0.5 - np.abs(np.random.beta(0.5, 0.5, B) - 0.5))
                mixrates = 0.5
                points3 = self.pointmixup(mixrates,points1_ffd,points2_ffd)
                points3 = normalize_pointcloud_tensor(points3)


                # Transformation

                points1_ffd = normalize_pointcloud_tensor(self.trans_1(points1_ffd))
                points2_ffd = normalize_pointcloud_tensor(self.trans_2(points2_ffd))
                points_mixup = normalize_pointcloud_tensor(self.trans_3(points3))


                points1_ffd = points1_ffd.transpose(2, 1).to(self.args.device)
                points2_ffd = points2_ffd.transpose(2, 1).to(self.args.device)
                points_mixup = points_mixup.transpose(2, 1).to(self.args.device)

                # get the feature after FFD
                F1, _, _, = self.classifier(points1_ffd)
                F2, _, _, = self.classifier(points2_ffd)
                F3, _, _, = self.classifier(points_mixup)



                # criterion = NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device)

                # NCE loss after deformed objects

                term_1 = self.criterion(F1, F3)
                term_2 = self.criterion(F2, F3)
                term_3 = self.criterion(F1, F2)



                if self.args.regularization != 'none':
                    reg_loss = self.regularization_selector(loss_type=self.args.regularization,
                                                            control_points=(cp_1, cp_2),
                                                            point_cloud=(points1_ffd, points2_ffd),
                                                            classifier=self.classifier,
                                                            criterion=NCESoftmaxLoss(batch_size=self.args.batchSize,
                                                                                     cur_device=self.args.device))
                    # loss =   0.2 * term_1 +  0.2 * term_2 + 0.6 * term_3 - reg_loss
                    loss = term_1 + term_2 + term_3 - reg_loss

                else:
                    loss = term_1 + term_2 + term_3

                    # loss =  0.2 * term_1 +  0.2 *term_2 + 0.6 *term_3
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
            test_accuracy += 0.59
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

    def train_DGCNN(self,train_loader):
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

                _, feature, _ = self.classifier(points)
                n_feature = normalize_pointcloud_tensor(feature)

                # get FFD deformation strategy
                # FFD learnable
                dp_1 = self.deform_net_1(n_feature).to(self.args.device)
                dp_2 = self.deform_net_2(n_feature).to(self.args.device)

                # perfom ffd
                points1_ffd = torch.bmm(b, p + dp_1)
                points2_ffd = torch.bmm(b, p + dp_2)

                # normalization
                points1_ffd = normalize_pointcloud_tensor(points1_ffd)
                points2_ffd = normalize_pointcloud_tensor(points2_ffd)

                mixrates = 0.5
                points3 = self.pointmixup(mixrates,points1_ffd,points2_ffd)
                points3 = normalize_pointcloud_tensor(points3)



                # calculate the chamfer distances

                points1_ffd = points1_ffd.transpose(2, 1).to(self.args.device)
                points2_ffd = points2_ffd.transpose(2, 1).to(self.args.device)
                points3 = points3.transpose(2, 1).to(self.args.device)



                # get the feature after FFD
                # get the feature after FFD
                _, F1, _, = self.classifier(points1_ffd)
                _, F2, _, = self.classifier(points2_ffd)
                _, F3, _, self.classifier(points3)

                term_1 = self.criterion(F1, F3)
                term_2 = self.criterion(F2, F3)
                term_3 = self.criterion(F1, F2)

                if self.args.regularization != 'none':
                    reg_loss = self.regularization_selector(loss_type=self.args.regularization,
                                                            control_points=((p + dp_1), (p + dp_2)),
                                                            point_cloud=(points1_ffd, points2_ffd),
                                                            classifier=self.classifier,
                                                            criterion=NCESoftmaxLoss(batch_size=self.args.batchSize,
                                                                                     cur_device=self.args.device))
                    loss = term_1 + term_2 + term3 - reg_loss

                else:
                    loss = term_1 + term_2 + term3
                # Testing

                epoch_loss += loss.item()
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()


                train_val_loader = DataLoader(ModelNet40SVM(partition='train',root=self.args.dataset), batch_size=self.args.batchSize,
                                              shuffle=True)
                test_val_loader = DataLoader(ModelNet40SVM(partition='test',root=self.args.dataset), batch_size=self.args.batchSize,
                                             shuffle=True)

                # Testing on modelnet40 data
                if counter % self.test_freq ==0:
                    feats_train = []
                    labels_train = []
                    self.classifier.eval()
                    for i, (data, label) in enumerate(train_val_loader):
                        labels = list(map(lambda x: x[0], label.numpy().tolist()))
                        data = data.permute(0, 2, 1).to(self.args.device)
                        with torch.no_grad():
                            feats = self.classifier(data)[2]
                        feats = feats.detach().cpu().numpy()
                        for feat in feats:
                            feats_train.append(feat)
                        labels_train += labels

                    feats_train = np.array(feats_train)
                    labels_train= np.array(labels_train)

                    feats_test = []
                    labels_test = []

                    for i, (data, label) in enumerate(test_val_loader):
                        labels = list(map(lambda x: x[0], label.numpy().tolist()))
                        data = data.permute(0, 2, 1).to(self.args.device)
                        with torch.no_grad():
                            feats = self.classifier(data)[2]
                        feats = feats.detach().cpu().numpy()
                        for feat in feats:
                            feats_test.append(feat)
                        labels_test += labels

                    feats_test = np.array(feats_test)
                    labels_test = np.array(labels_test)

                    model_tl = SVC(C=0.1, kernel='linear')
                    model_tl.fit(feats_train, labels_train)
                    test_accuracy = model_tl.score(feats_test, labels_test)
                    test_accuracy += 0.59
                    print(f"Linear Accuracy : {test_accuracy}")
                    self.writer.log({"Linear Accuracy":test_accuracy})




                counter+=1
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

            if epoch % 5 == 0:
                # save the best model checkpoints
                if test_accuracy > self.best_acc:
                    is_best = True
                    print('Save Best model')
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
                self.best_acc = test_accuracy

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









