import json
import glob, os
import shutil
import argparse
import cv2
from random import sample

parser = argparse.ArgumentParser()
parser.add_argument('--set_dir', default='/home/io18230/Desktop/F1627simple_140_B/001', type=str)
#parser.add_argument('--set_dir', default='/home/io18230/0Projects/keras-retinanet-master/path/demo_3images/images/val', type=str)
args = parser.parse_args()

#for im in sorted(os.listdir(args.set_dir)):

n = len(os.listdir(args.set_dir))
number_image = round(n/2)
all = list(range(1, int(number_image+1)))
print (number_image)
kk = len(all)
#print (kk)

#select 40%
p_40 = int(round(number_image*0.4))
s_40 = sample(all, p_40)
s_40.sort()
print (len(s_40))

#select 30%
p_30 = int(round(p_40*0.75))
s_30 = sample(s_40, p_30)
s_30.sort()
print (s_30)
print (len(s_30))

p_20 = int(round(p_30*2/3))
s_20 = sample(s_30, p_20)
s_20.sort()
print (s_20)
print (len(s_20))

p_10 = int(round(p_20*0.5))
s_10 = sample(s_20, p_10)
s_10.sort()
print (s_10)
print (len(s_10))

t_40 = list(set(all)-(set(s_40)))
t_30 = list(set(all)-(set(s_30)))
t_20 = list(set(all)-(set(s_20)))
t_10 = list(set(all)-(set(s_10)))

print (t_10)
print (len(t_30),len(t_20),len(t_10))

with open('/home/io18230/Desktop/val_40%.txt', 'w+') as f4:
    for ip in s_40:
        f4.write(str("%05d" % ip))
        f4.write('\n')
with open('/home/io18230/Desktop/val_40%_train.txt', 'w+') as t4:
    for ip in t_40:
        t4.write(str("%05d" % ip))
        t4.write('\n')

with open('/home/io18230/Desktop/val_30%.txt', 'w+') as f3:
    for ip in s_30:
        f3.write(str("%05d" % ip))
        f3.write('\n')
with open('/home/io18230/Desktop/val_30%_train.txt', 'w+') as t3:
    for ip in t_30:
        t3.write(str("%05d" % ip))
        t3.write('\n')


with open('/home/io18230/Desktop/val_20%.txt', 'w+') as f2:
    for ip in s_20:
        f2.write(str("%05d" % ip))
        f2.write('\n')
with open('/home/io18230/Desktop/val_20%_train.txt', 'w+') as t2:
    for ip in t_20:
        t2.write(str("%05d" % ip))
        t2.write('\n')


with open('/home/io18230/Desktop/val_10%.txt', 'w+') as f1:
    for ip in s_10:
        f1.write(str("%05d" % ip))
        f1.write('\n')
with open('/home/io18230/Desktop/val_10%_train.txt', 'w+') as t1:
    for ip in t_10:
        t1.write(str("%05d" % ip))
        t1.write('\n')