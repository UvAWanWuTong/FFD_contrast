from __future__ import print_function
import os
from utils.utils import save_config_file,save_checkpoint
from utils.criterion import  NCESoftmaxLoss

from utils.utils import save_config_file,save_checkpoint,normalize_pointcloud_tensor
from tqdm.auto import tqdm
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from Data.dataset import  ModelNet40SVM
from sklearn.svm import SVC
import numpy as np


from strategy.FFD_contrast import FFD_contrast

class FFD_learnable_contrast(FFD_contrast):
    def __init__(self,*args,**kwargs):
        super(FFD_learnable_contrast,self).__init__(*args,**kwargs)
        self.classifier = self.model_list[0].train()
        self.deform_net_1 = self.model_list[1].train()
        self.deform_net_2 = self.model_list[2].train()
        self.writer.watch(self.classifier)
        self.test_freq = 1000
        self.sigmoid = nn.Sigmoid()


    def train(self,train_loader):
        for epoch in tqdm(range(self.args.nepoch)):
            """ contrastive learning """
            counter = 0
            epoch_loss = 0
            for data in  train_loader:
                (b,p) = data
                b = b.to(self.args.device)
                p = p.to(self.args.device)

                points = torch.bmm(b, p)
                points = points.transpose(2, 1).to(self.args.device)
                self.optimizer.zero_grad()


                feature, _, _ = self.classifier(points)
                n_feature = normalize_pointcloud_tensor(feature)

                # get FFD deformation strategy
                # FFD learnable
                dp_1 = self.sigmoid(self.deform_net_1(n_feature)).to(self.args.device)
                dp_2 = self.sigmoid(self.deform_net_2(n_feature)).to(self.args.device)


                # perfom ffd
                points1_ffd = torch.bmm(b,p+dp_1)
                points2_ffd = torch.bmm(b,p+dp_2)

                # normalization
                points1_ffd = normalize_pointcloud_tensor(points1_ffd)
                points2_ffd = normalize_pointcloud_tensor(points2_ffd)


                points1_ffd = points1_ffd.transpose(2, 1).to(self.args.device)
                points2_ffd = points2_ffd.transpose(2, 1).to(self.args.device)

                # get the feature after FFD
                F1, _, _, = self.classifier(points1_ffd)
                F2, _, _, = self.classifier(points2_ffd)


                if self.args.regularization != 'none':
                    reg_loss = self.regularization_selector(loss_type=self.args.regularization,control_points=((p+dp_1),(p+dp_2)),point_cloud=(points1_ffd,points2_ffd),classifier=self.classifier,criterion=NCESoftmaxLoss(batch_size=self.args.batchSize, cur_device=self.args.device))
                    loss = self.criterion(F1, F2) - reg_loss

                else:
                    loss = self.criterion(F1, F2)




                epoch_loss  += loss.item()

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                if self.args.regularization != 'none':
                    self.writer.log({
                        "train loss": loss.item(),
                        "reg loss": reg_loss.item(),
                        "Train epoch": epoch,
                        "Learning rate": self.scheduler.get_last_lr()[0],

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
                    'state_dict': self.classifier.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=checkpoint_name,file_dir=self.args.save_path)
                self.min_loss = loss


                #save deform net
                deform_net_name = 'deform_net_1.pth.tar'
                save_checkpoint({
                    'state_dict': self.deform_net_1.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=deform_net_name, file_dir=self.args.save_path,save_deform=True)
                self.min_loss = loss


                deform_net_name = 'deform_net_2.pth.tar'
                save_checkpoint({
                    'state_dict': self.deform_net_2.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=is_best, filename=deform_net_name, file_dir=self.args.save_path,save_deform=True)
                self.min_loss = loss

    def train_PointNet(self,train_loader):
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
                n_feature = normalize_pointcloud_tensor(projection_feature )

                # get FFD deformation strategy
                # FFD learnable
                dp_1 = self.deform_net_1(n_feature).to(self.args.device)
                dp_2 = self.deform_net_2(n_feature).to(self.args.device)


                dp_1  = self.sigmoid(dp_1)
                dp_2  = self.sigmoid(dp_2)

                # perfom ffd
                points1_ffd = torch.bmm(b, p + dp_1)
                points2_ffd = torch.bmm(b, p + dp_2)

                # normalization
                points1_ffd = normalize_pointcloud_tensor(points1_ffd)
                points2_ffd = normalize_pointcloud_tensor(points2_ffd)

                points1_ffd = points1_ffd.transpose(2, 1).to(self.args.device)
                points2_ffd = points2_ffd.transpose(2, 1).to(self.args.device)

                # get the feature after FFD
                F1, _, _, = self.classifier(points1_ffd)
                F2, _, _, = self.classifier(points2_ffd)

                if self.args.regularization != 'none':
                    reg_loss = self.regularization_selector(loss_type=self.args.regularization,
                                                            control_points=((p + dp_1), (p + dp_2)),
                                                            point_cloud=(points1_ffd, points2_ffd),
                                                            classifier=self.classifier,
                                                            criterion=NCESoftmaxLoss(batch_size=self.args.batchSize,
                                                                                     cur_device=self.args.device))
                    loss = self.criterion(F1, F2) - reg_loss

                else:
                    loss = self.criterion(F1, F2)
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

                points1_ffd = points1_ffd.transpose(2, 1).to(self.args.device)
                points2_ffd = points2_ffd.transpose(2, 1).to(self.args.device)

                # get the feature after FFD
                _, F1, _, = self.classifier(points1_ffd)
                _, F2, _, = self.classifier(points2_ffd)

                if self.args.regularization != 'none':
                    reg_loss = self.regularization_selector(loss_type=self.args.regularization,
                                                            control_points=((p + dp_1), (p + dp_2)),
                                                            point_cloud=(points1_ffd, points2_ffd),
                                                            classifier=self.classifier,
                                                            criterion=NCESoftmaxLoss(batch_size=self.args.batchSize,
                                                                                     cur_device=self.args.device))
                    loss = self.criterion(F1, F2) - reg_loss

                else:
                    loss = self.criterion(F1, F2)
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





















