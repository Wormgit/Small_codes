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



# track for backpropgation    设置方法二选一
x = torch.ones(2, 2, requires_grad=True)  #如果.requires_grad 设置为 True ,它将开始track在其上的所有操作(这样就可以利利用用链式法则进行行行梯度传播了了)。
#x = torch.ones(2, 2)     #way 2 修改属性:
#x.requires_grad_(True)

print(x)
print(x.grad_fn)  #直接创建的称为叶子子节点,叶子子节点对应的 grad_fn 是 None
y = x + 2
print(y)
print(x.is_leaf, y.is_leaf)



# calculate gradient
z = y * y * 3
out = z.mean()
print(z)
print(out)
out.backward() # = out.backward(torch.tensor(1.))
print(x.grad)  #out 关于 x 的梯度 (导数)



#grad在反向传播过程中是累加的(accumulated),这意味着每一一次运行行行反向传播,所以反向传播之前需把梯度清零。
out2 = x.sum()
out2.backward()
print(x.grad)  # 不清零就累加上次的
out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)
print('\n\n')



# 只有scaler 张亮才能backpro
xx = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
yy = 2 * xx
zz = yy.view(2, 2)
print(zz)
#现在 zz 不是一个标量,所以在调用 backward 时需要传入一个和 zz 同形的权重向量进行加权求和得到标量。
# 不允许张量对张量求导,只允许标量对张量求导,求导结果是和自变量同形的张量。
# v 的值不影响结果,因为对v求导留下的是z.真聪明
v = torch.tensor([[1.0, 1], [1, 0.001]], dtype=torch.float)
print (torch.sum(zz * v))
zz.backward(v) #先计算 标量 = torch.sum(y * v) 然后求 标量 对自自变量量 x 的导数
print(xx.grad)   #xx.grad 是和 x 同形的张量量



# 不回传部分计算
x1 = torch.tensor(1.0, requires_grad=True)
y1 = x1 ** 2
with torch.no_grad():   # 有关的梯度是不会回传的 不被追踪 /使用 detach
    y2 = x1 ** 3        # 在评估模型的时候很常用,因为在评估模型时,我们并不不需要计算可训练参数的梯度。
y3 = y1 + y2
print('x1', x1, x1.requires_grad)
print('y2', y2, y2.requires_grad) # False
print('y3', y3, y3.requires_grad) # True
y3.backward()
print(x1.grad)  # 所以结果是2 不是5
print('\n\n')



#如果我们想要修改 tensor 的数值,但是又又不不希望被 autograd 记录(即不不会影响反向传播),
#那么我么可以对 tensor.data 进行操作
x11 = torch.ones(1,requires_grad=True)
print(x11)
print(x11.data) # 还是一一个tensor,已经是独立立于计算图之外
y11 = 2 * x11
x11.data *= 100 # 只改变了了值,不不会记录在计算图,所以不不会影响梯度传播
y11.backward()
print(x11) # 更更改data的值也会影响tensor的值
print(x11.grad)