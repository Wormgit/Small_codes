import json
import glob, os
import shutil

f = open("/home/io18230/Desktop/will.txt")

content=f.readlines()
for line in content:
    image_id = line.replace('\n','')
    image_id = image_id.replace('            ', '')
    image_id = image_id.replace('        ]', ']\n')

    with open('/home/io18230/Desktop/Output.txt','a+') as f2:
        f2.write(image_id)

f2.close()