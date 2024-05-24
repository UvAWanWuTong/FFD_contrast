import torch
from torch import nn
import numpy as np
import os
import numpy as np
import itertools
import math, random
random.seed = 42
import numpy as np
import open3d as o3d

import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt

import pygem
print(pygem.__version__)

from plyfile import PlyData
import scipy.spatial.distance
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pygem import FFD
from visualize import pcshow,pc_show_multi,visualize_rotate,pcwrite
from point_utils import *
from bernsetin import *
from path import Path
import torch
from torch import nn
from model.model import Deform_Net,PointNetCls,Contrastive_PointNet
import re
from emd__ import emd_module
import argparse
EMD = emd_module.emdModule()


def pointmixup(mixrates, xyz1, xyz2):
    # mix_rate = torch.tensor(mixrates).to(self.args.device).float()
    # mix_rate = mix_rate.unsqueeze_(1).unsqueeze_(2)
    # mix_rate_expand_xyz = mix_rate.expand(xyz1.shape).to(self.args.device)
    xyz1 = torch.tensor(xyz1).unsqueeze(0)
    xyz2 = torch.tensor(xyz2).unsqueeze(0)

    _, ass = EMD(xyz1, xyz2, 0.005, 300)
    ass = ass.cpu().numpy()
    xyz2 = xyz2[0][ass]
    xyz = xyz1 * (1 - mixrates) + xyz2 * mixrates

    return xyz.cpu().numpy()


def read_ply(f):
    plydata = PlyData.read(f)
    verts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    face = plydata['face']['vertex_index'].T

    return verts, face


def np_to_tensor(x):
    return torch.from_numpy(x.astype(np.float32))

def deform_point(tensor,classifier,deform_net1,deform_net2):
    classifier.eval()
    feats = classifier(tensor)   
    norm_feat = torch.nn.functional.normalize(feats[0], p=2.0, dim = 1)
    dp1 = deform_net1(norm_feat)
    dp1 = dp1.detach().numpy()


    dp2 = deform_net2(norm_feat)
    dp2 = dp2.detach().numpy()

    return dp1[0],dp2[0]

def FreeFormDeformation(point_set, num_points=60, points=5):
        def random_move(ffd):
            # randomly move the control points of  the ffd
            ffd = ffd
            point = np.random.randint(0, len(ffd.array_mu_x), size=[3])
            ffd.array_mu_x[point[0], point[1], point[2]] = np.random.uniform(0.5, 1.5)
            ffd.array_mu_y[point[0], point[1], point[2]] = np.random.uniform(0.5, 1.5)
            ffd.array_mu_z[point[0], point[1], point[2]] = np.random.uniform(0.5, 1.5)
            return ffd

        # def random_move(ffd):
        #         # randomly move the control points of  the ffd
        #         ffd = ffd

        #         point = np.random.randint(0, len(ffd.array_mu_x), size=[len(ffd.array_mu_x)])
        #         ffd.array_mu_x[point] = np.random.uniform(0.5, 1.5)
        #         ffd.array_mu_y[point] = np.random.uniform(0.5, 1.5)
        #         ffd.array_mu_z[point] = np.random.uniform(0.5, 1.5)
        #         return ffd
        # initialize the 27 control points with box length of 2
        ffd = FFD([points, points, points])

        ffd.box_length = [2, 2, 2]
        for i in range(num_points):
            ffd = random_move(ffd)

        deformed_points = ffd(np.asarray(point_set) + 1)

        return deformed_points, ffd.control_points()


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Load models and dataset paths.")

    parser.add_argument('--deform_net1_path', type=str, default='deform_net_1.pth.tar',
                        help='Path to the first deform net model')
    parser.add_argument('--deform_net2_path', type=str, default='deform_net_2.pth.tar',
                        help='Path to the second deform net model')
    parser.add_argument('--classifier_path', type=str, default='best_model.pth.tar',
                        help='Path to the classifier model')
    parser.add_argument('--dataset_path', type=str, default='/home/wan/Datasets/ModelNet40', help='Path to the dataset')

    args = parser.parse_args()


    folder_name = 'Deformed_Objects'
    if not os.path.exists(folder_name):
    # If it doesn't exist, create the folder
        os.mkdir(folder_name)
        print(f"Folder {folder_name} created successfully.")

    # Load the models using the parsed arguments
    deform_net1 = Deform_Net(in_features=128, out_features=(5 + 1) ** 3 * 3)
    deform_net1.load_state_dict(torch.load(args.deform_net1_path, map_location=torch.device('cpu'))['state_dict'],
                                strict=False)

    deform_net2 = Deform_Net(in_features=128, out_features=(5 + 1) ** 3 * 3)
    deform_net2.load_state_dict(torch.load(args.deform_net2_path, map_location=torch.device('cpu'))['state_dict'],
                                strict=False)

    classifier = Contrastive_PointNet()
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=torch.device('cpu'))['state_dict'],
                               strict=False)

    print("load model successfully")


    
    path = Path('/home/wan/Datasets/ModelNet40')

    data_set_path  =  '/home/wan/Datasets/ModelNet40'


    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
    # classes

    limit_count = 0

    for cls in folders:
        print(cls)
        if '.' in cls or limit_count>30:
            limit_count = 0
            continue
        # get current class folder
        cls_folder_name = cls
        if not os.path.exists(folder_name+'/'+cls_folder_name):
        # If it doesn't exist, create the folder
            os.mkdir(folder_name+'/'+cls_folder_name)
            print(f"Folder {cls_folder_name} created successfully.")

        else:
            continue


        # get all the files in the folder
        file_list = os.listdir(path/cls/'train')

        for file in file_list:
            with open(os.path.join(data_set_path+'/'+cls+'/train',file),'rb') as f:
            # with open(path/cls/'train'/file, 'rb') as f:
                data  = read_ply(f)
                verts, faces  = data 


            # get the berstin parameter, control points
            b,p,xyz= _calculate_ffd(np.array(verts),faces,n=5,n_samples=3072)
            
            # get origin 3d img
            origin = np.matmul(b,p)
            origin_tensor = np_to_tensor(np.array(origin)).unsqueeze(0)
            origin_tensor = origin_tensor.transpose(2, 1)

            # get deformed control points
            dp1,dp2 = deform_point(origin_tensor,classifier,deform_net1,deform_net2)

            # get the new 3d img
            new1 =  np.matmul(b,p+dp1)
            new1 = Normalize()(new1)


            new2 =  np.matmul(b,p+dp2)
            new2 = Normalize()(new2)


            # 这个地方放pointmixup

            mixrates = 0.5
            new3 = pointmixup(mixrates=mixrates,xyz1=new1,xyz2=new2)
            new3 = Normalize()(new3[0])


            file_name = file.split('.')[0]
            img_path = folder_name+'/'+cls_folder_name+'/'+file_name            
            # save img here
            pcwrite(img_path+'_deform1',*(new1).T)
            pcwrite(img_path+'_deform2',*(new2).T)
            pcwrite(img_path+'_mixup',*(new3).T)
            pcwrite(img_path,*(origin).T)



            limit_count +=1

            



            #create class folder
            #create img folder
            #


    

