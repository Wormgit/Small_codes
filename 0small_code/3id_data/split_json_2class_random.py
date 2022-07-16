import json
import glob, os
import shutil
import argparse
import cv2
from random import sample
import random

parser = argparse.ArgumentParser()
parser.add_argument('--set_dir', default='/home/io18230/Desktop/RGB/', type=str)
#parser.add_argument('--set_dir', default='/home/io18230/0Projects/keras-retinanet-master/path/demo_3images/images/val', type=str)
args = parser.parse_args()
sc ='/home/io18230/Desktop/'+'single_train_valid_test_splits.json'
cow = {}

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
    cow [im] = {}
    path = os.path.join(args.set_dir,im)
    list_of_file = os.listdir(path)

    n = len(list_of_file)
    other, train = data_split(list_of_file, ratio=0.2, shuffle=True)
    val, test = data_split(other, ratio=0.5, shuffle=True)

    cow[im]['test'] = test
    cow[im]['train']=train
    cow[im]['valid'] = val

with open(sc, 'w+') as f:
    json.dump(cow, f)




