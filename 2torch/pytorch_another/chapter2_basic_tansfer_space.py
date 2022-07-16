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



# build a tensor
x = torch.rand(5, 3)
x = torch.zeros(4, 3, dtype=torch.long)
x = torch.tensor([5.5, 3])

x = x.new_ones(5, 3, dtype=torch.float64)
x = torch.randn_like(x, dtype=torch.float)
print(x)
#获取形状 注意:返回的torch.Size其实就是一一个tuple, 支支持所有tuple的操作。
print(x.shape) #or print(x.size())



#加法操作
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))   # same effect
y.add_(x) # # adds x to y #save space
print(y)
#b = (a * a).sum()



#索引
y = x[0, :]
y += 1
print(y)
print(x[0, :]) # 源tensor也被改了了



#改变形状
y = x.view(15)
print (x)
z = x.view(-1, 5) # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
x += 1
print (x)
print(y) #view() 返回的新tensor与源tensor共享内存(其实是同一一个tensor),也即更更改其中的一一个,另外一一个也会跟着改变。(view仅仅是改变了了对这个张量量的观察⻆角度)

# 真正新的副本 ( 即 不 共 享 内 存 )
# 使用用 clone 还有一一个好处是会被记录在计算图中,即梯度回传到副本时也会传到源 Tensor 。
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)



#item() , 它可以将一个标量量 Tensor 转换成一一个Python number:
x = torch.randn(1)
print(x)
print(x.item())  # only for a number not vector nor matrix



#广播机制 形状不同就最小公倍数 useful for b(神经元的加的权重)
x = torch.arange(1, 3).view(1, 2) # start to end -1
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)



#2.2.4 运算的内存开销 索引、 view 是不不会开辟新内存的,而而像 y = x + y 这样的运算是会新开内存的,然后
#将 y 指向新内存
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x    # y[:] = y + x
print(id(y) == id_before) # False
#如果想指定结果到原来的 y 的内存,使用用前面面介绍的索引来进行行行替换操作。
#我们把 x + y 的结果通过 [:] 写进 y 对应的内存中。
#我们还可以使用用运算符全名函数中的 out 参数或者自自加运算符 += (也即 add_() )达到上述效果,例如
# torch.add(x, y, out=y) # y += x, y.add_(x)
# print(id(y) == id_before) # True



#2.2.5 TENSOR 和NUMPY相互转换
# torch.tensor()进行行行数据拷⻉ 返回的 Tensor 和原来的数据不不再共享内存, 下面例子共享.
# numpy() 和 from_numpy() 共享内存
a = torch.ones(5)
b = a.numpy()
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)


#NumPy数组转 Tensor   #在CPU上的 Tensor (除了 CharTensor )都支支持与NumPy数组相互转换。
a = np.ones(5)
b = torch.from_numpy(a) #共享内存
c = torch.tensor(a) #不共享内存



#2.2.6 TENSOR ON GPU 只有在PyTorch GPU版本上才会执行
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    Tensor
    x = x.to(device)   # # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # # to()还可以同时更更改数据类型
