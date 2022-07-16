#!/usr/bin/env python
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import numpy as np  # notes are in main_test
import os, cv2, time, glob, shutil, copy, random
import matplotlib.pyplot as plt
import sys, math, warnings
import argparse
import json

import torchvision
import torchvision.datasets as dset
import torchvision.utils
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.utils.data as Data

from torch.nn import init
from torch import optim
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from IPython import display

#使用用PyTorch可以更更简洁地实现模型
#torch.utils.data 模块提供了了有关数据处理理的工工具,
#torch.nn 模块定义了了大大量量神经网网络的层,
#torch.nn.init 模块定义了了各种初始化方方法,
#torch.optim 模块提供了了模型参数初始化的各种方方法。


#采用用矢矢量量计算,以提升计算效率  广播机制比手动设置慢
a = torch.ones(1000)
b = 10
c = 10*torch.ones(1000)
startt = time.time()
d = a + b    # c
print(time.time() - startt)


# do a simple back propagation
#Genearte dataset:
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4] ; true_b = 4.2  # ground truth
#features = torch.from_numpy(np.random.normal(0, 1, (num_examples,num_inputs))) # 均值0 , 标准差1  # 这个报错
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01,size=labels.size()))
#labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float) # labels 用两个都可
print(features[0], labels[0])




# plot data(generated gt)
def use_svg_display():
    display.set_matplotlib_formats('svg')# 用用矢矢量量图显示
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize # 设置图的尺寸寸
# 在../d2lzh_pytorch里里里面面添加上面面两个函数后就可以这样导入入
#sys.path.append("..")
#from d2lzh_pytorch import *
# 本函数已保存在d2lzh包中方方便便以后使用用 ???????????
set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show()



batch_size = 10
dataset = Data.TensorDataset(features, labels)                  # 将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  # 随机读取小小批量量

for X, y in data_iter:
    print(X, y)
    break

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net) # 使用用print可以打印出网网络的结构

# 写法1
net = nn.Sequential(nn.Linear(num_inputs, 1))   # 此处还可以传入入其他层
# 写法二
net2 = nn.Sequential()
net2.add_module('linear', nn.Linear(num_inputs, 1))  # net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))]))

#通过 net.parameters() 来查看模型所有的可学习参数,此函数将返回一一个生生成器器。
for param in net.parameters():
    print(param)

#torch.nn 仅支支持输入入一一个batch的样本不不支支持单个样本输入入,如果只有单个样本,可使
#用 input.unsqueeze(0) 来添加一一维


#3.3.4 初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0) # 也可以直接修改bias的data:  net[0].bias.data.fill_(0)

#3.3.5 定义损失函数
loss = nn.MSELoss()

#3.3.6 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
# optimizer =optim.SGD([          # 如果对某个参数不不指定学习率,就使用用最外层的默认学习率
#     {'params': net.subnet1.parameters()}, # lr=0.03
#     {'params': net.subnet2.parameters(), 'lr': 0.01}
#     ], lr=0.03)
print(optimizer)
# 调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

#3.3.7 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零,等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
    
    
dense = net[0]
print (dense)
print(true_w, dense.weight)
print(true_b, dense.bias)