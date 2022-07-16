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
import torchvision.models  #包 含 常 用用 的 模 型 结 构 ( 含 预 训 练 模 型 ) , 例例 如 AlexNet 、 VGG
import torchvision.datasets as dset  #一一些加载数据的函数及常用用的数据集接口口;
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


mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
train=True, download=True, transform=transforms.ToTensor())
mnist_test =  torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))  #训练集中和测试集中的每个类别的图像数分别为6,000和1,000。因为有10个类别,所以训练集和测试集的样本数分别为60,000和
#10,000。

feature, label = mnist_train[0]
print(feature.shape, label) # Channel x Height X Width
#我们使用用了了 transforms.ToTensor() ,所以每个 像素的数值为[0.0, 1.0]的32位浮点数。
#feature 的尺寸寸是 (C x H x W) 的,而而不不是 (Hx W x C)。第一维是通道数,因为数据集中是灰度图像,所以通道数为1

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress',
    'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# plot data(generated gt)
def use_svg_display():
    display.set_matplotlib_formats('svg')# 用用矢矢量量图显示

def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里里里的_表示我们忽略略(不不使用用)的变量量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
    show_fashion_mnist(X, get_fashion_mnist_labels(y))

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