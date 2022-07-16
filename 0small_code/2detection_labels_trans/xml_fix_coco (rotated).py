import json, math
from xml.etree import ElementTree as et
import xml.dom.minidom as x
import glob, os
#remenber to delete contenct in dict{}, find a right place for it.
#pay atterntion to id_tep

adjust_angle = 1  # 0 - 180 degree
step3_head = 1
# using coco format now

#xml_path = '/home/io18230/0Projects/keras-retinanet-master/path/Detection_rota/RGB2020_detec/images/train'
xml_path = '/home/io18230/0Projects/keras-retinanet-master/path/Detection_rota/Test'
#xml_path = '/home/io18230/0Projects/keras-retinanet-master/path/Rotate_inbarn_rgb/images/train'

save_json_path = xml_path + '/instances_test.json' #

cow = {}
cow['info'] = {"description":"COW Dataset rotated-box--train","year":2020}
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
angle =0
cx =0
cy =0
w =0
h =0

xmin = 0
ymin = 0
xmax = 0
ymax = 0

xmls = sorted(glob.glob(os.path.join(xml_path, '*.xml')))
count = 0
head_centres=[]

for xml in xmls:
    #print (xml)
    tree = et.parse(xml)
    root = tree.getroot()
    img_dict = {}
    for children in root:
    #get info for 'images' in json
        if children.tag == 'filename':
            id_tep=xml.replace(xml_path+'/', '')
            id_tep=id_tep.strip('.xml')
            #img_dict['file_name'] = children.text+'.jpg'
            #img_dict['id'] = int(children.text.strip('.jpg'))
            img_dict['file_name'] = id_tep + '.jpg'
            img_dict['id'] = int(id_tep.strip('.jpg'))
        if children.tag == 'size':
            for child in children:
                if child.tag == 'width':
                    img_dict['width'] = int(child.text)
                elif child.tag == 'height':
                    img_dict['height'] = int(child.text)

    #get all head centerpoint and save them in head_center
        for gt in children.iter('object'):
            for c in gt:
                if c.tag == 'name':
                    if c.text == 'cow':  # so it is head
                        break
                if c.tag == 'robndbox':
                    for last in c.iterfind('cx'):
                        cx = round(float(last.text))
                    for last in c.iterfind('cy'):
                        cy = round(float(last.text))
                    head_centres.append((cx, cy))
                    print(img_dict['id'])
                if c.tag == 'bndbox':
                    #print (img_dict['id'] )
                    #print ('not using rotated boxes, does not matter')
                    for last in c.iterfind('xmin'):
                        xmin = round(float(last.text))
                    for last in c.iterfind('ymin'):
                        ymin = round(float(last.text))
                    for last in c.iterfind('xmax'):
                        xmax = round(float(last.text))
                    for last in c.iterfind('ymax'):
                        ymax = round(float(last.text))
                    cx = (xmin + xmax)/2
                    cy = (ymin + ymax) / 2
                    head_centres.append((cx, cy))


    #get info for 'annotations' in json
    for children in root:
        for g in children.iter('object'):
            ann_dict = {}
            box = []
            ann_dict['iscrowd'] = 0
            ann_dict['image_id'] = img_dict['id']

            for c in g:
                if c.tag == 'name':

                    if c.text == 'cow':
                        ann_dict['category_id'] = 1
                        count = count + 1
                        ann_dict['id'] = count
                    else:# c.text == 'head':
                        break
                    #else:
                        #ann_dict['category_id'] = 0
                if c.tag == 'robndbox':
                    for last in c.iterfind('cx'):
                        cx = round(float(last.text))
                    for last in c.iterfind('cy'):
                        cy = round(float(last.text))
                    for last in c.iterfind('w'):
                        w =  round(float(last.text))
                    for last in c.iterfind('h'):
                        h =  round(float(last.text))
                    for last in c.iterfind('angle'):
                        angle = round(float(last.text),3)

                    if angle > 3.142:
                        angle = angle - 3.142
                    if angle < -3.142:
                        angle = angle + 3.142

                    w_final = 0
                    h_final = 0
                    a_final = 0
                    if w < h:
                        #print(img_dict['id'])
                        h_trans = w
                        w_trans = h
                        a_trans = round(angle - 1.571,3)

                        w_final = w_trans
                        h_final = h_trans
                        a_final = a_trans
                    else:
                        w_final = w
                        h_final = h
                        a_final = angle
                    if adjust_angle:
                        if a_final > 0:
                            a_final = round(a_final - 3.142,3)
                            #print (a_final)  #angle is between 0 to -3.14 now, next adjust them to 3.14 to -3.14.

                    if step3_head:
                        for iii in range(len(head_centres)):
                            cx_head_one_of_all, cy_head_one_of_all = head_centres[iii]
                            if (h_final/2)**2 > (cx-cx_head_one_of_all)**2+(cy-cy_head_one_of_all)**2:
                                cx_head = cx_head_one_of_all
                                cy_head = cy_head_one_of_all
                                xxxx = cx_head_one_of_all-cx
                                if xxxx == 0: #incase they are in the same line
                                    xxxx = 1

                                a_head = -(abs(math.atan((cy_head-cy) /xxxx)))

                                if cx_head >= cx and cy_head <= cy:
                                    a_head = a_head
                                elif cx_head <= cx and cy_head <= cy:
                                    a_head = -3.142-a_head
                                elif cx_head <= cx and cy_head >= cy:
                                    a_head = 3.142+a_head
                                elif cx_head >= cx and cy_head >= cy:
                                    a_head = -a_head

                                if a_head >  3.142:
                                    a_head = a_head - 3.142
                                if a_head < -3.142:
                                    a_head = a_head + 3.142

                                if  abs(a_final-a_head) >1.57:
                                    if a_final<0 and a_head>0:
                                        if (a_final+3.142*2 - a_head)<1.57:
                                            pass
                                            #print('special attention to 3.14 and -3.14')
                                            #print(ann_dict['image_id'])
                                        else:
                                            a_final = round(a_final + 3.142, 3)
                                    elif a_head<0 and a_final>0:
                                        if (a_head+3.142*2 - a_final)<1.57:
                                            pass
                                            #print('special attention to 3.14 and -3.14')
                                            #print(ann_dict['image_id'])
                                        else:
                                            a_final = round(a_final + 3.142, 3)
                                    else:
                                        #print('adjust angle of image')
                                        a_final = round(a_final + 3.142,3)
                                    if a_final > 3.142:
                                        a_final = a_final - 3.142
                        #print (a_final)

                    #ann_dict['bbox'] = [cx, cy, w_final, h_final, a_final]  # did not change the name for convinience
                    ann_dict['bbox'] = [cx-w_final/2, cy-h_final/2, w_final, h_final, a_final]   ##coco format
                    cow['annotations'].append(ann_dict)
                    ann_dict = {}
    head_centres.clear()
    cow['images'].append(img_dict)

with open(save_json_path, 'w+') as f:
    json.dump(cow, f)