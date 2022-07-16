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
folder = 'tes'
parser.add_argument('--set_dir', default='/home/io18230/Desktop/RGBDCows2020will/val_bp_remove4day/Identification/RGB/', type=str) #RGBDCows2020/Identification/RGB
args = parser.parse_args()

delete_folder=[]

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

movelist=[]

for im in sorted(os.listdir(args.set_dir)):
    path = os.path.join(args.set_dir,im)#,'0')
    list_of_file = os.listdir(path)

    for item in list_of_file:
        if item[0]!='i':
            m = item[5:10]
            assert m!='0'
            if m[0] != '0':
                print(111111111111111111)
            if m[1] == '3':
                if m[3] =='0' and m[4] in ['8','9']:
                    movelist.append(item)
                    pt = os.path.join('/home/io18230/Desktop/', folder, im)
                    makedirs(pt)
                    shutil.move(os.path.join(path, item), os.path.join(pt, item))
                if m[3] == '1':
                    movelist.append(item)
                    pt = os.path.join('/home/io18230/Desktop/', folder, im)
                    makedirs(pt)
                    shutil.move(os.path.join(path, item), os.path.join(pt, item))
        else:
            m = item[19:24]
            assert m!='0'
            if m[0] != '0':
                print(111111111111111111)
            #print(m[1],m[3])
            if m[1] == '3':
                if m[3] =='0' and m[4] in ['8','9']:
                    movelist.append(item)
                    pt = os.path.join('/home/io18230/Desktop/', folder, im)
                    makedirs(pt)
                    shutil.move(os.path.join(path, item), os.path.join(pt, item))
                if m[3] == '1':
                    movelist.append(item)
                    pt = os.path.join('/home/io18230/Desktop/', folder, im)
                    makedirs(pt)
                    shutil.move(os.path.join(path, item), os.path.join(pt, item))

