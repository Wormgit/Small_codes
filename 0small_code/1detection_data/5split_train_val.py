import glob, os
import shutil

path = '/home/io18230/Desktop/online_data/de'
save_json_path = '/home/io18230/Desktop/online_data/train'

c = 0
for x in sorted(os.listdir(path)):
#for x in xmls:

    if c%20 != 0:
        if x.endswith('.jpg'):
            name, ext = os.path.splitext(x)
            shutil.copy(os.path.join(path, x), os.path.join(save_json_path, x))
            shutil.copy(os.path.join(path, name+'.xml'), os.path.join(save_json_path, name+'.xml'))
    c=c+1
