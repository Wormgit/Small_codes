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
parser.add_argument('--set_dir', default='/home/io18230/0Projects/keras-retinanet-master/path/ID/1627video_on_server/20days/', type=str) #RGBDCows2020/Identification/RGB
args = parser.parse_args()

move_folder =[]

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


#################for video ####################
for folders in sorted(os.listdir(args.set_dir)):
    for img in sorted(os.listdir(args.set_dir+'/'+folders)):
        pa = os.path.join(args.set_dir, folders, img)
        img = cv2.imread(pa)
        h, w, _ = img.shape
        if w / h < 1.8:
            print(pa)
            # pt = os.path.join('/home/io18230/Desktop/', 'tooshort', img)
            # makedirs(pt)
            # shutil.move(os.path.join(path, item), os.path.join(pt, item)) #pt


#################for video ####################


#################for video ####################
# for im in sorted(os.listdir(args.set_dir)):
#     for sub in sorted(os.listdir(args.set_dir+'/'+im)):
#         path = os.path.join(args.set_dir,im, sub)
#
#         list_of_file = os.listdir(path)
#         for item in list_of_file:
#             pa = os.path.join(path,item)
#             img = cv2.imread(pa)
#             h, w, _ = img.shape
#             if w / h < 1.8:
#                 print(im, sub, item)
#                 pt = os.path.join('/home/io18230/Desktop/', 'tooshort', im)
#                 makedirs(pt)
#                 shutil.move(os.path.join(path, item), os.path.join(pt, item)) #pt

#################for video ####################



#################for detection teset ####################
# for im in sorted(os.listdir(args.set_dir)):
#     if len(im)> 10:
#         pa = os.path.join(args.set_dir,im)
#         img = cv2.imread(pa)
#         h, w, _ = img.shape
#         if w / h < 1.8:
#             print(im[3:8])
#             #shutil.copy(pa, os.path.join('/home/io18230/Desktop/check', im))  # pt
#             shutil.copy(os.path.join(args.set_dir,im[3:8]+'.jpg'), os.path.join('/home/io18230/Desktop/check', im[3:8]+'.jpg'))  # pt
#################for detection teset ####################