import json
import glob, os
import shutil

f = open("/home/io18230/Desktop/302.txt")

content=f.readlines()
txt_name = []
for line in content:
    if line !='\n':
        txt_name.append(line.rstrip('\n'))

folder_name = []
for item in os.listdir('/home/io18230/Desktop/Sub-levels/Identification/Videos'):
    folder_name.append(item)

for find in txt_name:
    the = find
    if the in folder_name:
        print(the)
m = 1