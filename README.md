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
cd lib
sh make.sh
```


### Preparation

Download the initial backbone weight from [Onedrive](https://1drv.ms/u/s!AjFGSP2CJWCwgXE38zYaQRR3a6L9?e=8FNHFg), and put it in the directory data/pretrained_model



Extract ActivityNet1.3 frames by FPS=3 following [R-C3D](https://github.com/VisionLearningGroup/R-C3D/blob/master/preprocess/activityNet/generate_frames.py), after that please put them in the directory ```dataset/activitynet13/train_val_frames_3/```, it ought to contain two folders: ```training, validation```.


The detail structure of the dataset is already splitted in our pickle file in ```./preprocess```. If you want to create your own dataset, you can follow [here](https://github.com/sunnyxiaohu/R-C3D.pytorch/blob/master/preprocess/activitynet/generate_roidb_training.py#L137) to create your own pickle file.



### Training

```
python main.py --bs 1 --gpus 0
```


### Evaluate our trained weight 

Firstly, download our trained weight from [Onedrive](https://1drv.ms/u/s!ArycXAIEda_Kcexadq6DPu0AF5o?e=fID2NN), and put the trained weight file ```best_model.pth``` in ```train_log/main```, then do the evaluation following the command:

```
python main.py --test
```

### Email for QA

Any question related to the repo, please send email to us: ```yangpengwan2016@gmail.com```.


### Acknowledgement

This repo is developed based on [https://github.com/sunnyxiaohu/R-C3D.pytorch](https://github.com/sunnyxiaohu/R-C3D.pytorch), thanks for their contribution.

### Citation

If you think our work is useful, please kindly cite our work.

```
@INPROCEEDINGS{YangECCV20,
        author = {Pengwan Yang and Vincent Tao Hu and Pascal Mettes and Cees G. M. Snoek},
        title = {Localizing the Common Action Among a Few Videos},
        booktitle = {European Conference on Computer Vision},
        month = {August},
        year = {2020},
        address = {Glasgow, UK},
      }
```




