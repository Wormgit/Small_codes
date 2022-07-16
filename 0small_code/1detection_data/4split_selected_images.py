import json
import glob, os
import shutil

f = open("/home/io18230/Desktop/will.txt")
path = '/home/io18230/Desktop/results_for_paer/04'
save_path = '/home/io18230/Desktop/3'

content=f.readlines()
for line in content:
    image_id = line.replace(path+'/','')
    image_id = image_id.replace('\n','')
    a = int(image_id)
    #if float(image_id) > 1:
    new_name = "%06d" % a
    new_name = new_name +'.jpg'

    shutil.copy(os.path.join(path, new_name), os.path.join(save_path, new_name))
print('Done')


