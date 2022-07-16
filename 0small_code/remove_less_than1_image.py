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
folder = '/home/io18230/Desktop/move_temp'
parser.add_argument('--set_dir', default='/home/io18230/Desktop/des/', type=str) #RGBDCows2020/Identification/RGB
args = parser.parse_args()

delete_folder=[]

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

# find folders contains only 1 image
for im in sorted(os.listdir(args.set_dir)):
    path = os.path.join(args.set_dir, im)#,'0')
    list_of_folder = os.listdir(path)
    list_of_folder.sort()


    for item in list_of_folder:
        foldername = os.path.join(path, item)
        list_of_file = os.listdir(os.path.join(path, item))
        if len(list_of_file) < 2:
            # print(foldername)
            delete_folder.append(foldername)


# move folders
makedirs(folder)
for item in delete_folder:
    last = item[-9:]
    shutil.move(item, folder+last)

print('moved', len(delete_folder), 'folders containing only 1 image')
m = 1



