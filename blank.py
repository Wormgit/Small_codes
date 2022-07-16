#!/usr/bin/env python
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import numpy as np  # notes are in main_test
import os, cv2, time, glob, shutil, copy, random
import matplotlib.pyplot as plt
import sys, math, warnings
import argparse
import json

#tf
#import keras
#import tensorflow as tf

# pytorch
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
print(time.strftime('%Y-%m-%d-%H:%M:%S'))

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

sys.path.insert(0, '../')
print(sys.path)

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.bin"

parser = argparse.ArgumentParser()
parser.add_argument("--score_th", default=0.3, type=float)
args = parser.parse_args()

print(sys.path)
print(__file__)
print(__package__)



name = 'Swaroop'
if name.endwith('Swa'):
    print('Yes, the string starts with "Swa"')
if name.find('war') != -1:
    print('Yes, it contains the string "war"')
#The find method is used to locate the position of the given substring within the string; find returns -1 if it is unsuccessful in finding the substring.

# Allow relative imports when being executed as script.
# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#     __package__ = "keras_retinanet.bin"


from PIL import Image


# delete the rest


l =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71308, 66527, 71064, 70111, 54285, 71188, 68554, 78086, 61849, 64509, 82896, 76582, 52392, 57281, 74231, 54807, 68366, 75773, 51928, 78701, 28850, 29653, 28612, 28113, 30535, 26207, 29174, 29910, 29671, 29650, 29107, 36125, 30394, 36145, 27173, 36362, 28145, 32568, 36080, 29217, 29024, 27527, 25432, 33897, 28535, 33205, 29702, 28538, 29203, 30817, 26495, 30870, 30173, 25342, 27153, 35784, 27739, 29419, 37434, 32198, 37557, 26983, 30121, 35675, 32358, 4660, 1432, 8843, 16254, 61263, 5047, 12287, 9757, 4071, 1430, 2832, 9078, 14917, 8228, 62538]

l.sort(reverse = True)

k = l[:int(len(l)/10)]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass


# read images
def resize_pad_image(image, target_scale=1):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = (int(iw*0.9), int(ih*0.9))  # 目标图像的尺寸
    scale = target_scale
    nw = int(iw * scale)
    image = image.resize((nw, nw), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('RGB', (iw, ih), (0,0,0))  # black
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((iw - w) // 2, (ih - h) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    return new_image

imagepath = r'/home/io18230/Desktop/1.png'
# 读取图片

image = Image.open(imagepath)
# 改变图片大小
new_image = resize_pad_image(image, target_scale = 0.9)  # 填充图像
new_image.show()
# 保存图片

new_image.save('/home/io18230/Desktop/new.png', quality=95)  # 'JPEG', quality=95可以省略

