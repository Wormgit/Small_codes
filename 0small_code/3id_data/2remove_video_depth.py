import json
import glob, os
import shutil
import argparse
import cv2
from random import sample
import random
import shutil

parser = argparse.ArgumentParser()
folder = 'temp_video2'
parser.add_argument('--set_dir', default='/home/io18230/Desktop/metric_learning/make data/n145', type=str) #RGBDCows2020/Identification/RGB
parser.add_argument('--video_folder', default='/home/io18230/Desktop/vi', type=str)
args = parser.parse_args()

delete_folder=[]

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

# move selected
for im in sorted(os.listdir(args.set_dir)):
    path = os.path.join(args.video_folder,im)
    pt = os.path.join('/home/io18230/Desktop/', folder, im)

    if os.path.exists(path):
        makedirs(pt)
        shutil.copy(os.path.join(path, 'RGB.avi'), os.path.join(pt, 'RGB.avi'))
    else:
        print(im)


