import Net_Dataloader
from Net_Dataloader import NetDataloader, set_train_val
from Net_Dataloader_ESTARFM import NetDataloader, set_train_val
#from 数据加载_三个时相 import NetDataloader, set_train_val
#from 数据加载1个时相 import NetDataloader, set_train_val
#from 微调数据加载 import NetDataloader, set_train_val
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from nets.unet_training import CE_Loss, Dice_loss, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.metrics import f_score
from 调整tanh为sigmoid import MyNet

#from Model_6band import MyNet
#from 加两层通道保证各通道占比并缩减通道数  import MyNet

from model_去掉F2 import MyNet


from model_land_12_modis_21 import MyNet  # 基本的 12波段 landsat  21波段modis的网络
from model_3时相 import MyNet
#from model_单时相 import MyNet
#from model_landsat_modis做一样的卷积次数 import MyNet
from model_去掉F2和F3 import MyNet


train_val_persent = 0.9  # train : val
epoch = 360
inputs_size = [256,256,8]

batch_size = 6 #6
NUM_CLASSES = 3 # 实际的类别数+1  因为还有个背景
band_num = 8
lr = 1e-4
device = torch.device('cuda:0')

# --------------------------------------------------------------------#

#   建议选项： 就是使用不同的损失
#   种类少（几类）时，设置为True
#   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
#   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
# ---------------------------------------------------------------------#

go_on = False # 是否接着训练
loss_history = LossHistory("E:/MyNet/loss_history_file")  # 训练权重的地址 并且有画损失函数的功能


# 加载数据######################################################
train_all = []  # 这是为了去换行符号 形成一个只带名字的列表 存放名字列表
total = 0  # 最后一行那个元素没有换行符 处理存放名字列表出现的错误
with open(r"E:\人工林识别论文\时空数据融合代码\txt_file\estarfm.txt", "r") as f:
    train_val = f.readlines()
for i in train_val:
    total += 1
    if total == len(train_val):
        train_all.append(i)
    else:
        train_all.append(i[0:-1])  # [1.tif,2.tif.......]
#################################################################


#model = Unet(num_classes=NUM_CLASSES, in_channels=inputs_size[-1], pretrained=pretrained).train()

model = MyNet(band_num=band_num).train()
net = torch.nn.DataParallel(model)
cudnn.benchmark = True
net = net.to(device)


if go_on:  # 是否接着训练 之前有没有训练权重
    model_path = r"E:\人工林识别论文\时空数据融合代码\model_data\estarfm_我的模型_加上ssim_landsat_modis_一样卷积.pth"  #
    # 加快模型训练的效率
    print('开始加载预训练文件')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('预训练模型加载完毕')


optimizer  = optim.Adam(model.parameters(), lr,weight_decay = 5e-4) #  ) 权重衰减
lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

ssmi_loss = False

for i in range(epoch):  # 第i个epoch

    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0

    train_lines, val_lines = set_train_val(train_all, train_val_persent)  # 重新分配 train 和 val

    batch_num_train = int(len(train_lines) / batch_size)  # 总共需要多少批次
    batch_num_val = int(len(val_lines) / batch_size)  # 验证集总共需要多少批次

    #for param in model.feature_net.parameters():
        #param.requires_grad = True

    net.train()
    with tqdm(total=batch_num_train, desc=f'Epoch {i + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
        for j in range(batch_num_train):  # 第j个batch
            batch = j  # 第几个batch
            if (batch + 1) * batch_size < len(train_lines):  # 别超过索引才能传进来训练
                # 传入的参数分别是： 所有名字的列表，第几个批次，批次数量，波段数，图像尺寸，图像类别个数
                train_RL_labe_mask_batch, train_modis, train_qa, labels  = NetDataloader(train_lines, batch, batch_size, band_num, image_size=inputs_size[0:2], num_classes=NUM_CLASSES).dataloader() # 读取数据
                # 自适应加权
                local_a = torch.zeros((1,),requires_grad=True)
                local_b = torch.zeros((1,), requires_grad=True)
                local_c = torch.zeros((1,), requires_grad=True)

                with torch.no_grad(): # 数据转tensor
                    train_RL_labe_mask_batch = torch.from_numpy(train_RL_labe_mask_batch).type(torch.FloatTensor)
                    train_modis = torch.from_numpy(train_modis).type(torch.FloatTensor)
                    train_qa = torch.from_numpy(train_qa).type(torch.FloatTensor)
                    labels = torch.from_numpy(labels).type(torch.FloatTensor)


                    labels = labels.view(-1,4,256,256)

                    train_RL_labe_mask_batch = train_RL_labe_mask_batch.to(device)

                    train_modis = train_modis.to(device)
                    train_qa = train_qa.to(device)

                    local_a = local_a.to(device)
                    local_b = local_b.to(device)
                    local_c = local_c.to(device)

                    labels = labels.to(device)



                optimizer.zero_grad()

                outputs = net(train_RL_labe_mask_batch,train_modis).to(device)


                outputs = outputs * train_qa  # 掩膜的有云的地方不参与计算

                loss = 0


                beta = [0.043**2, 0.059**2, 0.065**2, 0.176**2]
                for temp in range(4):

                    loss += F.mse_loss(outputs[:,temp,:,:], labels[:,temp,:,:], reduction='mean') / beta[temp]
                #######################自动加权
                print('Spectral_loss', loss)
                #loss = loss * local_a
                ndvi_train = (outputs[:, 3, :, :] - outputs[:, 2, :, :]) / (
                            outputs[:, 3, :, :] + outputs[:, 2, :, :] + 0.0000000001)


                # ndvi_train = torch.where(torch.isnan(ndvi_train), torch.full_like(ndvi_train,0), ndvi_train)


                ndvi_label = (labels[:, 3, :, :] - labels[:, 2, :, :]) / (
                            labels[:, 3, :, :] + labels[:, 2, :, :] + 0.0000000001)
                ndvi_label = torch.where(torch.isnan(ndvi_label), torch.full_like(ndvi_label, 0), ndvi_label)

                loss_ndvi = F.mse_loss(ndvi_label, ndvi_train, reduction='mean') / (0.5 ** 2)
                # loss_ndvi = loss_ndvi * local_b
                loss += loss_ndvi

                ########## NDBI 指数
                # ndbi_train = (outputs[:, 4, :, :] - outputs[:, 3, :, :]) / (
                #         outputs[:, 4, :, :] + outputs[:, 3, :, :] + 0.0000000001)
                # ndbi_label = (labels[:, 4, :, :] - labels[:, 3, :, :]) / (
                #         labels[:, 4, :, :] + labels[:, 3, :, :] + 0.0000000001)
                # ndbi_label = torch.where(torch.isnan(ndbi_label), torch.full_like(ndbi_label, 0), ndbi_label)
                # loss_ndbi = F.mse_loss(ndbi_label, ndbi_train, reduction='mean') / (0.5 ** 2)
                #
                # loss += loss_ndbi

                print('ndvi_loss', loss_ndvi)


                #print('loss',loss)
                #main_loss = CE_Loss(outputs, labels, num_classes=NUM_CLASSES) pspnet 的混合损失函数
                #loss = aux_loss + main_loss




                if i >= 20:

                    t_out = outputs #* 10000.0
                    t_label = labels #* 10000.0

                    list_true = t_label.view(1, -1).cpu().detach().numpy()

                    list_model = t_out.view(1, -1).cpu().detach().numpy()

                    mean_outputs = torch.mean(t_out)

                    mean_labels = torch.mean(t_label)

                    var_labels = torch.var(t_label)
                    var_outputs = torch.var(t_out)

                    # print(np.cov(list_true, list_model_result))
                    cov = np.cov(list_true, list_model)[0][1]  # 直接np.cov 得到的是协方差矩阵  取对角就是分别的协方差

                    c1 = (0.01 * 1.6384) ** 2
                    c2 = (0.03 * 1.6384) ** 2

                    ssmi = (2 * mean_outputs * mean_labels + c1) * (2 * cov + c2) / (
                            (mean_outputs ** 2 + mean_labels ** 2 + c1) * (var_labels + var_outputs + c2))

                    loss += (1 - ssmi) * 0.1  # * local_c
                    print('ssim_loss', (1-ssmi) * 0.1 )
                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = 0

                loss.backward() # 梯度计算
                optimizer.step() # 梯度更新  在一个batch里面

                total_loss += loss.item()


                total_f_score += 0

                pbar.set_postfix(**{'total_loss': total_loss / (batch + 1),
                                    'f_score': total_f_score / (batch + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)


    net.eval()
    print('\n开始验证')
    with tqdm(total=batch_num_val, desc=f'Epoch {i + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
        for j in range(batch_num_val):  # 第j个batch
            batch = j  # 第几个batch
            if (batch + 1) * batch_size < len(val_lines):  # 别超过索引才能传进来训练
                # 传入的参数分别是： 所有名字的列表，第几个批次，批次数量，波段数，图像尺寸，图像类别个数
                train_RL_labe_mask_batch, train_modis, train_qa, labels = NetDataloader(train_lines, batch, batch_size, band_num,
                                                                            image_size=inputs_size[0:2],
                                                                            num_classes=NUM_CLASSES).dataloader()  # 读取数据

                with torch.no_grad():  # 数据转tensor
                    train_RL_labe_mask_batch = torch.from_numpy(train_RL_labe_mask_batch).type(torch.FloatTensor)
                    train_modis = torch.from_numpy(train_modis).type(torch.FloatTensor)
                    train_qa = torch.from_numpy(train_qa).type(torch.FloatTensor)
                    labels = torch.from_numpy(labels).type(torch.FloatTensor)

                    labels = labels.view(-1, 4, 256, 256)

                    train_RL_labe_mask_batch = train_RL_labe_mask_batch.to(device)
                    train_modis = train_modis.to(device)
                    train_qa = train_qa.to(device)

                    labels = labels.to(device)



                    outputs = net(train_RL_labe_mask_batch,train_modis).to(device)
                    outputs = outputs * train_qa
                    loss = 0
                    beta = [0.043**2, 0.059**2, 0.065**2, 0.176**2]
                    ndvi_train = (outputs[:, 3, :, :] - outputs[:, 2, :, :]) / (
                                outputs[:, 3, :, :] + outputs[:, 2, :, :] + 0.0000000001)


                    ndvi_label = (labels[:, 3, :, :] - labels[:, 2, :, :]) / (labels[:, 3, :, :] + labels[:, 2, :, :] + 0.0000000001)

                    loss += F.mse_loss(ndvi_label, ndvi_train, reduction='mean') #/ (0.5 ** 2)  # 如果没有训练数据的预处理，这个是要加回来的
                    for temp in range(4):
                        loss += F.mse_loss(outputs[:,temp, :, :], labels[:,temp, :, :], reduction='mean') / beta[temp]


                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = 0

                    val_toal_loss += loss.item()
                    val_total_f_score += 0

                pbar.set_postfix(**{'total_loss': val_toal_loss / (batch + 1),
                                        'f_score': val_total_f_score / (batch + 1),
                                        'lr': get_lr(optimizer)})
                pbar.update(1)

    loss_history.append_loss(total_loss / (batch_num_train + 1), val_toal_loss / (batch_num_val + 1))
    print('Finish Validation')
    print('Epoch:' + str(i + 1) + '/' + str(epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (batch_num_train + 1), val_toal_loss / (batch_num_val + 1)))
    print('Saving state, iter:', str(i + 1))

    torch.save(model.state_dict(), 'E:/人工林识别论文/时空数据融合代码/model_data/model.pth')
    # torch.save(model.state_dict(), 'E:/人工林识别论文/时空数据融合代码/model_data/GX_2010-2011_%d-Total_Loss%.4f_val_loss%.4f.pth' % (
    # (i + 1), total_loss / (batch_num_train + 1),val_toal_loss/((batch_num_val + 1))))

    lr_scheduler.step() #梯度更新  在一个epoch里面