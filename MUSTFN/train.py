from Net_Dataloader import NetDataloader, set_train_val

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from MUSTFN import MyNet

#from 加两层通道保证各通道占比并缩减通道数  import MyNet


train_val_persent = 0.9  # train : val
epoch = 360
inputs_size = [256,256,8]
batch_size = 10 #6
band_num = 8
lr = 1e-4
device = torch.device('cuda:0')


ssmi_loss = False
go_on = False # 是否接着训练



# 加载数据######################################################
train_all = []  # 这是为了去换行符号 形成一个只带名字的列表 存放名字列表
total = 0  # 最后一行那个元素没有换行符 处理存放名字列表出现的错误
with open(r"E:\landsat数据的_rse超分辨率网络\txt_file\train2.txt", "r") as f:
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
    model_path = r""
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
                modis, landsat,  labels, masks  = NetDataloader(train_lines, batch, batch_size, band_num, image_size=inputs_size[0:2]).dataloader() # 读取数据
                #print(np.mean(labels[1, 1, :, :]))
                with torch.no_grad(): # 数据转tensor
                    modis = torch.from_numpy(modis).type(torch.FloatTensor)
                    landsat = torch.from_numpy(landsat).type(torch.FloatTensor)
                    labels = torch.from_numpy(labels).type(torch.FloatTensor)
                    masks = torch.from_numpy(masks).type(torch.FloatTensor)

                    labels = labels.view(-1,4,256,256)

                    modis = modis.to(device)
                    landsat = landsat.to(device)
                    labels = labels.to(device)
                    masks = masks.to(device)



                optimizer.zero_grad()

                outputs = net(landsat,modis).to(device) * masks



                loss = 0

                beta = [0.0300**2, 0.0500**2, 0.040**2, 0.2300**2]
                ndvi_train = (outputs[:,3,:,:] - outputs[:,2,:,:]) / (outputs[:,3,:,:] + outputs[:,2,:,:] + 0.0000000001)

                ndvi_label = (labels[:,3,:,:] - labels[:,2,:,:]) / (labels[:,3,:,:] + labels[:,2,:,:] + 0.0000000001)

                ndvi_label = torch.where(torch.isnan(ndvi_label), torch.full_like(ndvi_label, 0), ndvi_label)

                loss_ndvi = F.mse_loss(ndvi_label, ndvi_train, reduction='mean')

                loss += loss_ndvi


                for temp in range(4):

                    loss += F.mse_loss(outputs[:,temp,:,:], labels[:,temp,:,:], reduction='mean') / beta[temp]

                if i > 5:
                    ssmi_loss == True
                if ssmi_loss:
                    t_out = outputs * 10000.0
                    t_label = labels * 10000.0

                    list_true = t_label.view(1, -1).numpy()

                    list_model = t_out.view(1, -1).numpy()

                    mean_outputs = torch.mean(t_out)

                    mean_labels = torch.mean(t_label)

                    var_labels = torch.var(t_label)
                    var_outputs = torch.var(t_out)


                    # print(np.cov(list_true, list_model_result))
                    cov = np.cov(list_true, list_model)[0][1]  # 直接np.cov 得到的是协方差矩阵  取对角就是分别的协方差

                    c1 = (0.01 * 4096) ** 2
                    c2 = (0.03 * 4096) ** 2

                    ssim = (2 * mean_outputs * mean_labels + c1) * (2 * cov + c2) / (
                                (mean_outputs ** 2 + mean_labels ** 2 + c1) * (var_labels + var_outputs + c2))

                    loss += (1 - ssim)



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
                modis, landsat,  labels,  masks = NetDataloader(train_lines, batch, batch_size, band_num,
                                                                            image_size=inputs_size[0:2],
                                                                           ).dataloader()  # 读取数据

                with torch.no_grad():  # 数据转tensor
                    modis = torch.from_numpy(modis).type(torch.FloatTensor)
                    landsat = torch.from_numpy(landsat).type(torch.FloatTensor)
                    labels = torch.from_numpy(labels).type(torch.FloatTensor)
                    masks = torch.from_numpy(masks).type(torch.FloatTensor)

                    labels = labels.view(-1, 4, 256, 256)

                    modis = modis.to(device)
                    landsat = landsat.to(device)
                    labels = labels.to(device)
                    masks = masks.to(device)


                    outputs = net(landsat, modis).to(device)
                    outputs = outputs * masks

                    loss = 0
                    beta =  [0.0300**2, 0.0500**2, 0.0600**2, 0.2000**2]
                    ndvi_train = (outputs[:, 3, :, :] - outputs[:, 2, :, :]) / (
                                outputs[:, 3, :, :] + outputs[:, 2, :, :] + 0.0000000001)


                    ndvi_label = (labels[:, 3, :, :] - labels[:, 2, :, :]) / (labels[:, 3, :, :] + labels[:, 2, :, :] + 0.0000000001)

                    loss +=  torch.pow(F.mse_loss(ndvi_label, ndvi_train, reduction='mean'),1/2)
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


    print('Finish Validation')
    print('Epoch:' + str(i + 1) + '/' + str(epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (batch_num_train + 1), val_toal_loss / (batch_num_val + 1)))
    print('Saving state, iter:', str(i + 1))

    torch.save(model.state_dict(), 'E:/log_cloud/l8_%d-Total_Loss%.4f_val_loss%.4f.pth' % (
    (i + 1), total_loss / (batch_num_train + 1),val_toal_loss/((batch_num_val + 1))))

    lr_scheduler.step() #梯度更新  在一个epoch里面
