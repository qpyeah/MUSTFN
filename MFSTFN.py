import torch
import torch.nn as nn
from torchsummary import summary
import math
import torch.nn.functional as F


class DSconv(nn.Module): # 分组卷积
    def __init__(self, ch_in, ch_out, stride):
        super(DSconv, self).__init__()
        self.pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        self.conv = nn.Conv2d(ch_in,ch_out,kernel_size=3, stride=stride, padding=0,groups=ch_in, bias=False)  #  ch_out 要是group的整数倍
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1(nn.Module): # 1×1 卷积  调整维度
    def __init__(self, ch_in, ch_out):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(ch_in,ch_out,kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv(nn.Module): # 3×3 卷积  最终的融合
    def __init__(self, ch_in, ch_out,stride=1):
        super(Conv, self).__init__()
        self.pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        self.conv = nn.Conv2d(ch_in,ch_out,kernel_size=3, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Dconv(nn.Module): # 膨胀卷积
    def __init__(self, ch_in, ch_out, rate, stride=1):
        super(Dconv, self).__init__()
        self.pad = nn.ReplicationPad2d(padding=(rate, rate, rate, rate))
        self.conv = nn.Conv2d(ch_in,ch_out,kernel_size=3, stride=stride, padding=0, dilation= rate, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Attention(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=False)  # 对于注意力机制 这个参数是可以改的  就是中途通道的变化情况
        self.relu = nn.ReLU()
        self.conv1_2 = nn.Conv2d(ch_out, ch_out, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(ch_out, ch_in, 1, 1, 0, bias=False)
        self.activation = nn.Sigmoid() # 对[b,c,h,w] 的 c 进行 softmax

    def forward(self, x):
        global_feature1 = torch.mean(x,2,True) # 这两步操作就构成了全局平均池化
        global_feature2 = torch.mean(global_feature1,3,True) # 变成了 8，26，1，1

        x = self.conv1(global_feature2)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.conv2(x)
        x = self.activation(x) #

        return x # torch.mul(result,x) 就是注意力机制的结果


class Feature_net(nn.Module):  # 特征提取网络
    def __init__(self, band_num):
        # 第一个band_num 是输入进来的 参考合成landsat 和 palsar 的波段数

        super(Feature_net, self).__init__()
        self.deep_conv1 = Conv(ch_in=band_num, ch_out=band_num, stride=1) # 初步特征整合 和下面这一步一起
        self.deep_conv1_2 = Conv(ch_in=band_num, ch_out=band_num, stride=1)  # 初步特征整合 用来 残差连接  当作一个最初的特征
        self.deep_conv1_3 = Conv(ch_in=12, ch_out=12, stride=1)

        band_num = 20  #  modis 和 参考 landsat 融合
        self.conv_feature1 = nn.Sequential(
            Conv1(band_num, band_num*2),
            DSconv(band_num*2, band_num*2, stride=1),
            Conv1(band_num*2, band_num)
        )

        self.deep_conv2 = Dconv(40, 40*2, stride=2, rate=2)  #  60 128 128

        band_num = 40
        self.conv_feature2 = nn.Sequential(
            Conv1(band_num*2, band_num*4),
            Conv(band_num*4, band_num*4, stride=1),
            Conv1(band_num*4, band_num*2)
        )

        self.deep_conv3 = Dconv(band_num*2, band_num*4, stride=2, rate=2) #  120,64,64
        self.conv_feature3 = nn.Sequential(
            Conv1(band_num*4, band_num*8),
            Conv(band_num*8, band_num*8, stride=1),
            Conv1(band_num*8, band_num*4)
        )

        self._initialize_weights()

    def forward(self, x, x2):
        x = self.deep_conv1(x)

        x = self.deep_conv1_2(x)

        x2 = self.deep_conv1_3(x2)


        x = torch.cat((x, x2), dim=1)

        feature1 = torch.cat((x, self.conv_feature1(x)), dim=1)


        x = self.deep_conv2(feature1)
        feature2 = x + self.conv_feature2(x)

        x = self.deep_conv3(feature2)
        feature3 = x + self.conv_feature3(x)

        return feature1, feature2, feature3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Pool_net(nn.Module):
    def __init__(self,band_num):  # 60  128  128
        super(Pool_net, self).__init__()


        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(64, 64))
        self.conv2 = Conv1(band_num*2, band_num*4)

        self.pool3 = nn.AdaptiveAvgPool2d(output_size=(32, 32))
        self.conv3 = Conv1(band_num*2, band_num*4)

        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(16, 16))
        self.conv4 = Conv1(band_num*2, band_num*4)

        self.conv1 = Conv1(band_num * 4 * 3, band_num * 4)    # 120

    def forward(self, x):


        pool2 = self.pool2(x)
        pool2 = self.conv2(pool2)
        pool2 = F.interpolate(pool2, size=(256, 256), mode='bilinear', align_corners=True)

        pool3 = self.pool3(x)
        pool3 = self.conv3(pool3)
        pool3 = F.interpolate(pool3, size=(256, 256), mode='bilinear', align_corners=True)

        pool4 = self.pool4(x)
        pool4 = self.conv4(pool4)
        pool4 = F.interpolate(pool4, size=(256, 256), mode='bilinear', align_corners=True)


        x = torch.cat((pool2, pool3, pool4), dim=1)  # 120
        x = self.conv1(x)
        return x


class Dilate_net(nn.Module):
    def __init__(self,band_num):
        super(Dilate_net, self).__init__()

        self.d2 = Dconv(band_num*4, band_num*6, rate=3) # 不同的膨胀卷积
        self.d3 = Dconv(band_num*4, band_num*6, rate=5)
        self.d4 = Dconv(band_num*4, band_num*6, rate=7)

        self.conv1 = Conv1( band_num*6*3,  band_num*6)

    def forward(self, x):

        d2 = self.d2(x)
        d3 = self.d3(x)
        d4 = self.d4(x)

        x = torch.cat((d2, d3, d4), dim=1)
        x = self.conv1(x) #
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)

        return x

'''
(input 1) : landsat F1 and F3
(input x2) : modis M1 M2 and M3
'''



class MyNet(nn.Module):
    def __init__(self,band_num):
        super(MyNet, self).__init__()

        self.feature_net = Feature_net(band_num)
        self.pool = Pool_net(band_num=40) # 第二部分的特征
        self.dilate = Dilate_net(band_num=40) # 第三部分的特征

        self.attention2 = Attention(440, 150) #
        self.conv1 = Conv(440,128)
        self.conv2 = Conv(128, 64)

        #  6 个 要预测的landsat 波段
        self.conv3 = nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x, x2):


        feature1, feature2, feature3 = self.feature_net(x, x2)

        feature2 = self.pool(feature2)
        feature3 = self.dilate(feature3)

        result = torch.cat((feature1, feature2, feature3), dim=1)

        attention2 = self.attention2(result)
        result = torch.mul(result, attention2)

        result = self.conv1(result)
        result = self.conv2(result)
        result = self.conv3(result)

        result = self.sigmoid(result)
        return  result




def main(): # 测试维度变化的程序
    net = MyNet(band_num=8)

    device = torch.device('cuda:0')
    model = net.to(device)
    with torch.no_grad():
        summary(model, [(8, 256, 256),(12, 256, 256)])


if __name__ == '__main__':
    main()