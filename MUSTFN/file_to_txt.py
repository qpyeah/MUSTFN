import os
import random

random.seed(0)

segfilepath = r'D:\L8\trian\label_mask'   # label路径
saveBasePath = r"E:\landsat数据\txt_file" # 保存路径


all_file_list = os.listdir(segfilepath) #遍历文件夹下面的所有文件
#print(all_file_list)

all_file_name = []
for seg in all_file_list:
    if seg.endswith(".tif"): # 判断一下后缀
        all_file_name.append(seg)


num = len(all_file_name)
list = range(num) # range(0, num)
train = random.sample(list, num) # 把整个列表打乱[3,5,1,220,334,.....,]

train_txt = open(os.path.join(saveBasePath, 'train2.txt'), 'w')
total = 0
for i in train: # i 是一组随机打乱的序号
    total += 1
    if total < len(train):  # 保证最后一行不是换行符
        name = all_file_name[i][:-4] + '\n'
        train_txt.write(name)
    else:
        name = all_file_name[i][:-4]
        print(name)
        train_txt.write(name)

train_txt.close()

