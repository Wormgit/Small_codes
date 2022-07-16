# RGB adress.
# split val and test images to 2 folders by copying (2 class or 3?), no balck cattle

import json
import glob, os
import shutil
import argparse
import cv2
from random import sample
import random
import shutil


parser = argparse.ArgumentParser()
folder = 'RGBDCows2020w'
parser.add_argument('--set_dir', default='/home/io18230/Desktop/'+folder+'/Identification/RGB', type=str) #RGBDCows2020/Identification/RGB
args = parser.parse_args()

delete_folder=[]

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

for im in sorted(os.listdir(args.set_dir)):
    if im in ['054','069','073','173']:
        continue
    path = os.path.join(args.set_dir,im)
    list_of_file = os.listdir(path)

    n = len(list_of_file)
    val, test= data_split(list_of_file, ratio=0.3, shuffle=True)

    for item in val:
        pt = os.path.join('/home/io18230/Desktop/',folder, 'val/Identification/RGB', im, '0')
        makedirs(pt)
        shutil.copy(os.path.join(path, item),os.path.join(pt, item))

    for item in test:
        pt = os.path.join('/home/io18230/Desktop/',folder, 'test_2class', im)
        makedirs(pt)
        shutil.copy(os.path.join(path, item),os.path.join(pt, item))

    if len(val) < 2:
        print(f'Len of val of folder {im}: {len(val)}')
