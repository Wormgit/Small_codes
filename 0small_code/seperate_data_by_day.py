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
folder = 'des'
parser.add_argument('--set_dir', default='/home/io18230/Desktop/next/2class/farm1/', type=str) #RGBDCows2020/Identification/RGB
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

        else:
            m = item[19:24]
            assert m!='0'
            if m[0] != '0':
                print(111111111111111111)
            #print(m[1],m[3])

        movelist.append(item)
        m = m[0:2]+m[3:5]

        # if seperate by individual
        pt = os.path.join('/home/io18230/Desktop/', folder, m, im)
        # if only by date

        #pt = os.path.join('/home/io18230/Desktop/', folder, m)

        if not os.path.exists(pt):
            makedirs(pt)
        shutil.copy(os.path.join(path, item), os.path.join(pt, item))



