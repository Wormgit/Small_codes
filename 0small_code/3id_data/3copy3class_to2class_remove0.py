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
folder = 'tes'
parser.add_argument('--set_dir', default='/home/io18230/Desktop/test', type=str) #RGBDCows2020/Identification/RGB
args = parser.parse_args()

delete_folder=[]

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

for im in sorted(os.listdir(args.set_dir)):
    #path = os.path.join(args.set_dir,im)
    #list_sub_folder = os.listdir(path)
    #n = len(list_sub_folder)
    path = os.path.join(args.set_dir,im)
    list_of_file = os.listdir(path+'/0')

    for item in list_of_file:
        pt = os.path.join('/home/io18230/Desktop/', folder, im)
        makedirs(pt)
        shutil.copy(os.path.join(path+'/0', item), os.path.join(pt, item))  #source, destination
