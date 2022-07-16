import json
import glob, os, sys
import cv2

cow = {}
cow['info'] = {"description":"COW Dataset from will for species detection","time":'27/11/2019'}
cow['images'] = []
cow['annotations'] = []
cow['categories'] = [{"supercategory": "cow","id": 1,"name": "tian"}]

image_path ='/home/io18230/Desktop/images/val'
annotation_path  ='/home/io18230/Desktop/text_txt'
save_json_path = '/home/io18230/Desktop/instances_val.json'

count = 0

for id in sorted(os.listdir(image_path)):
    if id.endswith(".jpg") or id.endswith(".tif"):
        name, ext = os.path.splitext(id)
        img_dict = {}
        img_dict['file_name'] = id
        img_dict['id'] = int(name)
        #print(int(name))

        img = cv2.imread(os.path.join(image_path, id))
        img_dict['width'] = img.shape[1]
        img_dict['height'] = img.shape[0]

        cow['images'].append(img_dict)

        f = open(os.path.join(annotation_path, name + '.txt'))
        content = f.readlines()
        #print (content)
        for line in content:
            ann_dict = {}
            a = line.split(' ')
            #print (a)
            count = count + 1
            #print (count)
            ann_dict['iscrowd'] = 0
            ann_dict['image_id'] = img_dict['id']
            ann_dict['id'] = count
            ann_dict['category_id'] = 1
            scalex_centre = float(a[1])
            scaley_centre = float(a[2])
            scale_width = float(a[3])/2
            scale_height = float(a[4])/2

            xm = round((img.shape[1]-1)*(scalex_centre-scale_width))        # width * x scale
            if xm < 0:
                xm = 0
            ym = round((img.shape[0]-1)*(scaley_centre-scale_height))    # height * y scale
            if ym < 0:
                ym = 0
            xr = round((img.shape[1]-1)*(scale_width*2))
            if xr + xm > img.shape[1]-1:
                xr = img.shape[1]-1
            yr = round((img.shape[0]-1)*(scale_height*2))
            if yr + ym > img.shape[0] - 1:
                yr = img.shape[0]-1
            ann_dict['bbox'] = [xm, ym, xr, yr]
            #print(xm, ym, xr, yr)
            cow['annotations'].append(ann_dict)

with open(save_json_path, 'w+') as f:
    json.dump(cow, f)
