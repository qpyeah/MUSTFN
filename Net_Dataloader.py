'''

F: landsat
M: modis

label_mask = F2 + MASK(cloud:1, nocloud:0) , total 5bands
landsat_modis = train = 4bands F1 + 4bands F2 + 4bands M1 + 4bands M2 + 4 bands M3, total 20bands


'''
import os
import cv2
import numpy as np
from osgeo import gdal
from PIL import Image
from torch.utils.data.dataset import Dataset
import random
from torch.utils.data import DataLoader
import torch




def set_train_val(train, train_val_persent): # 按设定的比例重新分配 train和 val

    #random.shuffle(train) # 把这个注释掉就不会打乱train 和 val
    train_train = train[0:int(len(train) * train_val_persent)]
    train_val = train[int(len(train) * train_val_persent):-1]

    return train_train, train_val

def read_tif(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

    del dataset
    return im_proj, im_geotrans, im_data

class NetDataloader():  #
    def __init__(self,train_lines,batch,batch_size,band_num,image_size):
        super(NetDataloader,self).__init__()

        # train_lines 是从txt读到的 train 或者 val 的 label的名字
        self.train_lines = train_lines
        self.train_lines_num = len(train_lines)
        self.image_size = image_size

        self.batch_size = batch_size
        self.batch = batch
        self.index = 0
        self.band = band_num

    def dataloader(self):

        modis_batch =  np.zeros((self.batch_size, 12, self.image_size[0], self.image_size[1]))

        landsat_batch = np.zeros((self.batch_size, 8, self.image_size[0], self.image_size[1]))

        label_batch = np.zeros((self.batch_size, 4, self.image_size[0], self.image_size[1]))

        mask_batch =  np.zeros((self.batch_size, 1, self.image_size[0], self.image_size[1]))

        while self.index < self.batch_size: # 判断这个批次里面的元素是不是够了  够了就返回一个批次的 train 和 label

            name = self.train_lines[self.index + self.batch * self.batch_size] #

            landsat_modis_path = "D:/L8/trian/train" + '/' + name + ".tif"
            label_mask_path = "D:/L8/trian/label_mask" + '/' + name + ".tif"
            _, _, landsat_modis = read_tif(landsat_modis_path)
            _, _, label_mask = read_tif(label_mask_path)

            landsat = landsat_modis[0:8,:,:]
            modis = landsat_modis[8:20,:,:]

            
            label = label_mask[0:4,:,:]
            mask = 1-(label_mask[-1,:,:])

            modis_batch[self.index] =  modis * mask
            landsat_batch[self.index] =  landsat * mask
            label_batch[self.index] = label * mask
            mask_batch[self.index] = mask
            self.index += 1



        return modis_batch, landsat_batch , label_batch, mask_batch


