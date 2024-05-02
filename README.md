# FFD Contrast

This paper proposes a new method for point cloud data augmentation to address the current lack of point cloud data augmentation. We adopt a learnable Free Form Deformation (FFD) technique and utilize a deep neural deformation network (Deform Net) to change the shape of point cloud objects. In addition, we designed a new Mixup Module to generate a new point cloud object with all deformation features by mixing two deformed point cloud objects. Finally, we employ a Contrastive Learning (CL) approach to learn the optimal deformation representation of a point cloud. For a given point cloud object, we are able to generate four completely different deformation representations, which significantly expands the data volume and alleviates the scarcity of point cloud datasets. Through rich experimental validation, our method outperforms existing point cloud data enhancement techniques and achieves state-of-the-art performance in supervised learning, unsupervised learning, and sample less learning.
For more details, please check our [paper]().




### System Requirements

   * cuda=11.7
   * python 3.7
   * gcc=7.5.0
   * torch=1.13

### Package Requirements

```
conda create -n ffd_contrast  python=3.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands (emd loss):
```
cd utils/emd_ && python setup.py install 
```


### Preparation



Datasets are available [here](https://drive.google.com/drive/folders/1dAH9R3XDV0z69Bz6lBaftmJJyuckbPmR). Run the command below to download all the datasets (ShapeNetRender, ModelNet40, ScanObjectNN, ShapeNetPart) to reproduce the results.

```
cd Data

source download_data.sh
```

### Training

```
python train.py --model pointnet --dataset Data/ --nepoch 100 --dataset_type shapenet --lr 0.001 --decay 0.8 --step_size 2000 --batchSize 64 --ffd_points_axis 5 --task_type mixup --structure 3layer --feature_size 128 --regularization chamfer
```


### Evaluate our trained weight 

Firstly, download our trained weight from [Onedrive](https://1drv.ms/u/s!ArycXAIEda_Kcexadq6DPu0AF5o?e=fID2NN), and put the trained weight file ```best_model.pth``` in ```train_log/main```, then do the evaluation following the command:

```
 python evaluate.py --dataset Data/ --nepoch=150 --dataset_type modelnet40 --lr 0.003 --batchSize 64 --step_size=2000 --decay=0.8 --model model_path
```

### Email for QA

Any question related to the repo, please send email to us: ```1353099226why@gamil.com```.


### Acknowledgement

This repo is developed based on [https://github.com/MohamedAfham/CrossPoint.git](https://github.com/MohamedAfham/CrossPoint.git), thanks for their contribution.

### Citation

If you think our work is useful, please kindly cite our work.






