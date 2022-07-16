import json
import glob, os
import shutil
import argparse
import cv2
from random import sample
import random
import shutil
# RGB adress.

parser = argparse.ArgumentParser()
parser.add_argument('--set_dir', default='/home/io18230/Desktop/RGBDCows2020w/val/Identification/RGB', type=str) #RGBDCows2020/Identification/RGB
args = parser.parse_args()
sc ='/home/io18230/Desktop/'+'single_train_valid_test_splits.json'
cow = {}
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

def move_files(delete_folder, dc='/home/io18230/Desktop/move_to',):
    #makedirs(dc)
    for items in delete_folder:
        path = os.path.join(args.set_dir,items[0],items[1])
        new = os.path.join(dc,items[0])
        makedirs(new)
        shutil.move(path, new)



for im in sorted(os.listdir(args.set_dir)):
    cow [im] = {}
    path = os.path.join(args.set_dir,im)

    list_sub_folder = os.listdir(path)
    n = len(list_sub_folder)

    for inter in list_sub_folder:
        cow[im][inter] = {}
        path_sub = (os.path.join(path, inter))
        list_of_file = sorted(os.listdir(path_sub))
        n = len(list_of_file)
        train, other= data_split(list_of_file, ratio=0, shuffle=True) #train: other
        val, test = data_split(other, ratio=1, shuffle=True)
        if len(val) < 2:
            print(f'Len of val of folder {im},{inter}: {len(val)}, delete this in json')
            delete_folder.append([im,inter])
            del cow[im][inter] #delete in json.
            continue
        cow[im][inter]['test'] = test
        cow[im][inter]['train']=train
        cow[im][inter]['valid'] = val

    # if int(im)==5: #stop at folder xx
    #     break

move_files(delete_folder)  #move  subfolder

for im in sorted(os.listdir(args.set_dir)): # move black folder
    path = os.path.join(args.set_dir,im)
    list_sub_folder = os.listdir(path)
    if len(list_sub_folder) == 0:
        print('11')
        #shutil.move(path, '/home/io18230/Desktop/move_to')


with open(sc, 'w+') as f:
    json.dump(cow, f)



