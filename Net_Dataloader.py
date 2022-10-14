# 此函数用于返回某一种网络需要用到的某批次的 train 和 label label_onehot
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
    #print('看看是不是打乱了：',train_val[-1])
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
    def __init__(self,train_lines,batch,batch_size,band_num,image_size,num_classes):
        super(NetDataloader,self).__init__()

        # train_lines 是从txt读到的 train 或者 val 的 label的名字
        self.train_lines = train_lines
        self.train_lines_num = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.batch = batch
        self.index = 0
        self.band = band_num

    def dataloader(self):




        landsat_batch = np.zeros((self.batch_size, 12, self.image_size[0], self.image_size[1]))

        landsat = np.zeros((12, self.image_size[0], self.image_size[1]))

        modis=  np.zeros((21, self.image_size[0], self.image_size[1]))
        modis_batch =  np.zeros((self.batch_size, 21, self.image_size[0], self.image_size[1]))
        ###################################### ESTARFM 不需要这个mask 因为数据是全的
        mask =  np.zeros((  self.image_size[0], self.image_size[1]))
        #mask = np.ones((self.image_size[0], self.image_size[1]))
        mask_batch =  np.zeros((self.batch_size,  1, self.image_size[0], self.image_size[1]))

        label = np.zeros((6, self.image_size[0], self.image_size[1]))
        label_batch = np.zeros((self.batch_size, 6, self.image_size[0], self.image_size[1]))



        while self.index < self.batch_size: # 判断这个批次里面的元素是不是够了  够了就返回一个批次的 train 和 label

            name = self.train_lines[self.index + self.batch * self.batch_size] #


            land_lable_mask_file = "D:/ESTARFM/train/landsat" + '/' + name + ".tif"
            modis_file = 'D:/ESTARFM/train/modis' + '/' + name + ".tif"
            label_file = 'D:/ESTARFM/train/label' + '/' + name + ".tif"
            mask_file = 'D:/ESTARFM/train/mask' + '/' + name + ".tif"

            try:
                _, _, land_lable_mask_data = read_tif(land_lable_mask_file) # [c,w,h]
                _,_, modis_data = read_tif(modis_file)
                _,_, label_data = read_tif(label_file)
                _, _, mask = read_tif(mask_file)
                landsat[0:6,:,:] =  land_lable_mask_data[0:6]
                landsat[6:12, :, :] = land_lable_mask_data[6:12]
                modis =  modis_data


                label = label_data[0:6,:,:]


                landsat_batch[self.index] =  landsat * mask
                label_batch[self.index] =  label * mask
                modis_batch[self.index] = modis * mask
                mask_batch[self.index] = mask

                self.index += 1


            except:
                self.index += 1


        return landsat_batch, modis_batch, mask_batch, label_batch   #  train_qa_batch 只是为了控制 真正的测试和真正有云的部分不加参反应


def main():  # 知识测试程序
    epoch = 1
    image_size = [256, 256]
    batch_size = 4
    NUM_CLASSES = 20
    band_num = 3
    # 全局变量
    with open(r"F:\pytorch学习\my_net\train2.txt", "r") as f:
        train_lines = f.readlines()
    train_all = []  # 这是为了去换行符号 形成一个只带名字的列表
    total = 0  # 最后一行那个元素没有换行符
    for i in train_lines:
        total += 1
        if total == len(train_lines):
            train_all.append(i)
        else:
            train_all.append(i[0:-1])  # [1.tif,2.tif.......]

    train_val_persent = 0.9  # train : val
    ''''''
    for i in range(epoch): # 第i个epoch
        train_lines, val_lines = set_train_val(train_all, train_val_persent) # 重新分配 train 和 val
        print('有元素', len(train_lines))
        batch_num = int(len(train_lines) / batch_size) # 总共需要多少批次
        print('共需要：{0}个批次'.format(batch_num))
        for j in range(batch_num):  # 第j个batch
            batch = j  # 第几个batch
            if (batch+1)*batch_size < len(train_lines): # 别超过索引
                # 传入的参数分别是： 所有名字的列表，第几个批次，批次数量，波段数，图像尺寸，图像类别个数
                trains,labels,labels_onehot = NetDataloader(train_lines,batch,batch_size,band_num,image_size, NUM_CLASSES).dataloader()
                print('第{0}个epoch,第{1}个batch'.format(i,j))




if __name__ == '__main__':
    main()
