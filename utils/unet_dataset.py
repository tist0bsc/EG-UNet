import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from osgeo import gdal


def read_txt(path):
    ims, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split()
            ims.append(im)
            labels.append(label)
    return ims, labels

def read_label(filename):
    dataset=gdal.Open(filename)    #打开文件
 
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize  #栅格矩阵的行数
 
    # im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    # im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))

    del dataset 
    return im_data

def read_tiff(filename,train=True):
    dataset=gdal.Open(filename)    #打开文件
 
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize  #栅格矩阵的行数
 
    # im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    # im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))

    if train:
        im_data[1,...]= im_data[1,...]*255/1375
        im_data[2,...]= im_data[2,...]*255/1583
        im_data[3,...]= im_data[3,...]*255/1267
        im_data[4,...]= im_data[4,...]*255/2612
        im_data[0,...]= im_data[0,...]*255/122
    else:
        im_data[1,...]= im_data[1,...]*255/1375
        im_data[2,...]= im_data[2,...]*255/1583
        im_data[3,...]= im_data[3,...]*255/1267
        im_data[4,...]= im_data[4,...]*255/2612
        im_data[0,...]= im_data[0,...]*255/122
    del dataset 
    return im_data

def class_7(filename):
    label = np.array(read_tiff(filename))
    label_7 = label
    for i in range(len(label)):
        for j in range(len(label[i])):
            if label[i][j] in range(0,3):
                label_7[i][j]=0
            elif label[i][j] in range(3,7):
                label_7[i][j]=1
            elif label[i][j] in range(7,11):
                label_7[i][j]=2
            elif label[i][j] in range(11,13):
                label_7[i][j]=3
            elif label[i][j] in range(13,16):
                label_7[i][j]=4
            elif label[i][j] in range(16,19):
                label_7[i][j]=5
            elif label[i][j] == 19:
                label_7[i][j]=6
    return label_7


class UnetDataset(Dataset):
    def __init__(self, txtpath, transform,train=True):
        super().__init__()
        self.ims, self.labels = read_txt(txtpath)
        self.transform = transform
        self.train=train

    def __getitem__(self, index):
        root_dir = ''
        im_path = os.path.join(root_dir,self.ims[index])
        label_path = os.path.join(root_dir,self.labels[index])
        if_train=self.train

        image = read_tiff(im_path,if_train)
        image = np.array(image)
        image = np.transpose(image,(1,2,0))
        image = transforms.ToTensor()(image)
        image = image.to(torch.float32).cuda()
        image = self.transform(image).cuda()
        #20类
        label = torch.from_numpy(np.asarray(read_label(label_path), dtype=np.int32)).long().cuda()

        return image, label,label_path

    def __len__(self):
        return len(self.ims)