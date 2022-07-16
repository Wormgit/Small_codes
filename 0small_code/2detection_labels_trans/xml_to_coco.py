import json
from xml.etree import ElementTree as et
import xml.dom.minidom as x
import glob, os
#remenber to delete contenct in dict{}, find a right place for it.
number = ''

#xml_path = '/home/io18230/Desktop/'+number+'/images/train'
xml_path ='/home/io18230/0Projects/keras-retinanet-master/path/COWs/cross_val/split/7/images/test'
save_json_path = 'instances_val.json'

#sc = '/home/io18230/Desktop/'+number+'/annotations/'+save_json_path
sc ='/home/io18230/Desktop/'+save_json_path
cow = {}
cow['info'] = {"description":"COW Dataset--CORRECTED","year":'2020_AUG',"type":'10fold_include val'}
cow['images'] = []
cow['annotations'] = []
cow['categories'] = [{"supercategory": "cow","id": 1,"name": "xin"}]
box = []
xm = 0
ym = 0
xr = 0
yr = 0
xx = 0
yx = 0

xmls = sorted(glob.glob(os.path.join(xml_path, '*.xml')))
count = 0

for xml in xmls:
    tree = et.parse(xml)
    root = tree.getroot()
    img_dict = {}
    for children in root:
        #get info for 'images' in json
        if children.tag == 'filename':
            #img_dict['file_name'] = children.text
            img_dict['id'] = int(children.text.strip('.jpg'))
            img_dict['file_name'] = '%06d' % img_dict['id']+'.jpg'    # will's data miss .jpg sometimes in file_name area
        if children.tag == 'size':
            for child in children:
                if child.tag == 'width':
                    img_dict['width'] = int(child.text)
                elif child.tag == 'height':
                    img_dict['height'] = int(child.text)

    #get info for 'annotations' in json
        for g in children.iter('object'):
            ann_dict = {}
            box = []
            ann_dict['iscrowd'] = 0
            count = count + 1
            print (count)
            ann_dict['image_id'] = img_dict['id']
            ann_dict['id'] = count

            for c in g:
                if c.tag =='name':
                    if c.text == 'cow':
                        ann_dict['category_id'] = 1
                    else:
                        ann_dict['category_id'] = 0

                if c.tag == 'bndbox':
                    for last in c.iterfind('xmin'):
                        xm = int(last.text)
                    for last in c.iterfind('xmax'):
                        xx = int(last.text)
                    for last in c.iterfind('ymin'):
                        ym = int(last.text)
                    for last in c.iterfind('ymax'):
                        yx = int(last.text)

                    if xx > img_dict['width']-1:
                        xx = img_dict['width'] - 1
                    if yx > img_dict['height']-1:
                        yx = img_dict['height'] - 1

                    w = xx - xm
                    h = yx - ym

                    if xm < 0:
                        xm = 0
                    if ym < 0:
                        ym = 0

                    ann_dict['bbox'] = [xm, ym, w, h]
                    #print(xm, ym, w, h)
                    cow['annotations'].append(ann_dict)
                    ann_dict = {}
    cow['images'].append(img_dict)

with open(sc, 'w+') as f:
    json.dump(cow, f)