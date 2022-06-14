import os
from osgeo import gdal
import cv2 as cv

from MUSTFN import MyNet
from torch import nn
import numpy as np
import torch


def read_tif(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

    del dataset
    return im_proj, im_geotrans, im_data


def write_img(filename, im_proj, im_geotrans, im_data, im_height):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数

    # 创建文件
    im_bands = 4 ######################### 这里要改  保存的文件是几个波段
    im_height = im_height
    im_width = im_height
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


im_height = im_width = 256
model_path = r'E:\log_cloud\l8_147-Total_Loss0.0910_val_loss0.0731.pth'



landsat_modis_path = r'D:\L8\test\test'
file_all = os.listdir(landsat_modis_path)
#print(file_all)


net = MyNet(8)
net =net.eval()
state_dict = torch.load(model_path)
net.load_state_dict(state_dict)
net = nn.DataParallel(net)
net = net.cuda()


for file in file_all:

    landsat_modis = os.path.join(landsat_modis_path + '/' + file)
    proj, geotrans, landsat_modis_img = read_tif(landsat_modis)
    landsat = landsat_modis_img[0:8,:,:]
    modis = landsat_modis_img[8:20,:,:] #/ 10000.0



    modis = np.reshape(modis, (1, 12, 256, 256))
    landsat = np.reshape(landsat, (1, 8, 256, 256))

    with torch.no_grad():
        landsat = torch.from_numpy(landsat).type(torch.FloatTensor)
        modis = torch.from_numpy(modis).type(torch.FloatTensor)
        landsat = landsat.cuda()
        modis = modis.cuda()
        pr = net(landsat,modis)

        #print(pr.shape)
        pr = pr.cpu().numpy()

        pr = np.reshape(pr, (4, 256, 256))


    #r_image.save(r'G:\temp\%s'
    #r_image.show()
    write_img((r'D:\L8\result\%s'% file), proj, geotrans, pr,im_height)
