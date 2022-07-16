# RGB adress.

import json
import glob, os
import shutil
import argparse
import cv2
from random import sample
import random
import shutil


parser = argparse.ArgumentParser()
des_folder = '/home/io18230/Desktop/des'
parser.add_argument('--set_dir', default='/home/io18230/Desktop/next/2class/farm1', type=str) #RGBDCows2020/Identification/RGB
args = parser.parse_args()

delete_folder=[]

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

for im in sorted(os.listdir(args.set_dir)):
    current_path = os.path.join(args.set_dir, im)
    des_path = os.path.join(des_folder, im + '/0')
    list_of_file = os.listdir(current_path)
    makedirs(des_path)
    for item in list_of_file:
        shutil.copy(os.path.join(current_path, item), os.path.join(des_path, item))  #source, destination
