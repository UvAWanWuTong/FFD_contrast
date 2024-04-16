import torch.utils.data as data
import os
os.chdir(os.path.dirname(__file__))
import os.path
import torch
import sys
# from pygem import FFD
from utils.ffd_utils import  *
from plyfile import PlyData
from utils.sampler import Normalize,RandomSampler
from utils.utils import np_to_tensor
import glob
import h5py



def load_ply(file_name: str,
             with_faces: bool = False,
             with_color: bool = False) -> np.ndarray:
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]
    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val



def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../ModelNet40/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


def load_shapenet_data(root):
    all_filepath = []
    for cls in glob.glob(root+'/*'):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs

    return all_filepath


class Contrastive_ShapeNet(data.Dataset):
    def __init__(self,root,  npoints=1024,
                 split='train',
                 FFD = True,
                 ffd_points_axis=3,
                 ffd_control= 6):
        self.data = load_shapenet_data(root+'ShapeNet')
        self.ffd_points_axis = ffd_points_axis
        self.ffd_control= ffd_control
        self.ffd = FFD
        self.npoints = npoints
        self.root = root
        self.split = split
        self.fns = []

    def __getitem__(self, index):
        plydata = PlyData.read(self.data[index])
        point_set = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        point_set = RandomSampler(self.npoints)(point_set)
        point_set = Normalize()(point_set)
        b, p = calculate_ffd(points=point_set, n=self.ffd_points_axis)
        b = np_to_tensor(b)
        p = np_to_tensor(p)
        return (b, p)

    def __len__(self):
        return len(self.data)



class Contrastive_ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=1024,

                 split='train',
                 data_augmentation=False,
                 FFD = True,
                 ffd_points_axis=3,
                 ffd_control= 6):

        self.ffd_points_axis = ffd_points_axis
        self.ffd_control= ffd_control
        self.ffd = FFD
        self.npoints = npoints
        self.root = root+'ModelNet40'
        self.split = split
        self.data_augmentation = data_augmentation
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())
            f.close()

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../ModelNet40/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])
            f.close()

        print(self.cat)
        self.classes = list(self.cat.keys())
    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
            f.close()
        point_set = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T

        # point_set = PointSampler(self.npoints)((verts, face))
        point_set = RandomSampler(self.npoints)(point_set)
        point_set = Normalize()(point_set)



        b,p= calculate_ffd(points =point_set,n=self.ffd_points_axis)




        b = np_to_tensor(b)
        p = np_to_tensor(p)


        # point_set1 = torch.from_numpy(point_set1.astype(np.float32))
        # point_set2 = torch.from_numpy(point_set2.astype(np.float32))


        return (b,p)


    def __len__(self):
        return len(self.fns)



class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=False):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../ModelNet40/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        verts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T


        point_set = RandomSampler(self.npoints)(verts)
        point_set = Normalize()(point_set) #normolize


        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)


def load_ScanObjectNN(partition):
    BASE_DIR = 'data/ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')

    return data, label

def load_modelnet_data(root,partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(root, 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label



class ModelNet40SVM(data.Dataset):
    def __init__(self,root, num_points=1024, partition='train'):
        self.data, self.label = load_modelnet_data(root+'modelnet40_ply_hdf5_2048',partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNNSVM(data.Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]


    if dataset == 'modelnet':
        d= Contrastive_ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

    if dataset == 'svmmodelnet':
        d = ModelNet40SVM(root=datapath)
        print(len(d))
        print(d[0])
    if dataset == 'shapenet':
        d = Contrastive_ShapeNet(root=datapath)
        print(len(d))

    dataloader = torch.utils.data.DataLoader(d,batch_size=8,shuffle=True,num_workers=4,drop_last=True)
    data_loader_iter = iter(dataloader)
    b,p = next(data_loader_iter)

    print(b,p)








