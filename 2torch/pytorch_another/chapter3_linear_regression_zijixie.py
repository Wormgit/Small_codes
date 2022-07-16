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
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from IPython import display


#采用用矢矢量量计算,以提升计算效率  广播机制比手动设置慢
a = torch.ones(1000)
b = 10
c = 10*torch.ones(1000)
startt = time.time()
d = a + b    # c
print(time.time() - startt)


# do a simple back propagation
x_train = torch.rand(100)
y_train = x_train * 2 + 3 # w = 2, b = 3, y = 2 * x + 3
w = torch.tensor([0.0],requires_grad = True)
b = torch.tensor([0.0],requires_grad = True)

def loss_func(y_true: torch.Tensor, y_pre: torch.Tensor):
    square = (y_true - y_pre) ** 2
    return square.mean()

def train():
    lr = 0.015
    for i in range(400):
        y_pre = x_train * w + b

        loss = loss_func(y_train, y_pre)
        if i % 20 == 0:
            print("Iter: %d, w: %.4f, b: %.4f, training loss: %.4f" % (i, w.item(), b.item(), loss.item()))
        loss.backward()
        # 更新
        w.data -= w.grad * lr
        b.data -= b.grad * lr

        w.grad.data.zero_()
        b.grad.data.zero_()

train()




# do a simple back propagation in pytorch style
class SimpleLinear:
    def __init__(self):
        self.w = torch.tensor([0.0], requires_grad=True)
        self.b = torch.tensor([0.0], requires_grad=True)

    def forward(self, x):
        y = self.w * x + self.b
        return y

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, x):
        return self.forward(x)

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for para in self.parameters:
            para.data -= para.grad * self.lr

    def zero_grad(self):
        for para in self.parameters:
            para.grad.data.zero_()

model = SimpleLinear()
opt = Optimizer(model.parameters(), lr=0.3)

for epoch in range(10):
    output = model(x_train)
    loss = loss_func(y_train, output)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print('Epoch {}, loss is {:.4f}'.format(epoch, loss.item()))






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


def data_iter(batch_size, features, labels):  # 抽取10个indice, 通过incice 找到tensor数据
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一一次可能不不足足一一个batch
        if len(j)<10:
            print(1)
        yield features.index_select(0, j), labels.index_select(0,j)

batch_size = 10
for X, y in data_iter(batch_size, features, labels):  # batch x1,x2 y
    print(X, y)
    break

# initialazation 
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def linreg(X, w, b):
    return torch.mm(X, w) + b  #线性回归的矢矢量量计算
def squared_loss(y_hat, y): # 返回的是向量, 另外, pytorch里里里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里里里更更改param    时用用的param.data

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
# 训练模型一一共需要num_epochs个迭代周期
# 在每一一个迭代周期中,会使用用训练数据集中所有样本一次(假设样本数能够被批量大小整除)。X
# 和y分别是小小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小小批量量X和y的损失
        l.backward()  # 小小批量量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用用小小批量量随机梯度下降迭代模型参数
        w.grad.data.zero_() # 不不要忘了了梯度清零
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

#3.2.7