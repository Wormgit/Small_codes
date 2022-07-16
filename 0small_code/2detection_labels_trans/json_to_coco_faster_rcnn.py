import sys,math
import os
import json
from xml.etree.ElementTree import ElementTree, Element


dir_name = os.path.abspath(os.path.dirname(__file__))
libs_path = os.path.join(dir_name, '..', 'libs')
sys.path.insert(0, libs_path)

fold = 9

json_path = '/home/io18230/Downloads/detections/Faster-RCNN/'
file_name = 'fold-{}_to_be_evaluated.json'.format(fold)
annotation_file = json_path + file_name
dataset = json.load(open(annotation_file, 'r'))

def create_node(tag, content,node):
    element = Element(tag)
    element.text = content
    node.append(element)

c = 1
last_name = 'any'
results = []
ann = []
rank = []
difficult = 0
dts = dataset['preds']
gts = dataset['GT']

# for dt in dts:
#     rank_t = dt['filename'].strip('.jpg')
#     rank.append(int(rank_t))
#     rank.sort()
# print (rank)


for dt in dts:
    name = dt['filename'] #.strip('.jpg')
    s = dt['conf']
    if s<0.5:
        print (s)
    x1 = dt['x1']
    y1 = dt['y1']
    x2 = dt['x2']
    y2 = dt['y2']

    xe = (x1 + x2)/ 2
    ye = (y1 + y2)/ 2
    w =  x2 - x1
    h =  y2 - y1

    ttt = gts [name]

    image_result = {
        'file_name': name,  # can be sheild when calculate map
        'image_id': int(name.strip('.jpg')),
        'category_id': int(1),  # labels_to_names[label],######################
        'id': c,
        'score': float(s),
        'bbox': [x1, y1, w, h],  # (x1,y1,w,h)
        "width": ttt['img_w'],
        "height": ttt['img_h'],
        'ignore': 0,  # give ignore mark of those < ratio (inside image/ the detection box)
    }
    c = c + 1
    results.append(image_result)


json.dump(results, open('/home/io18230/Desktop/ij/pre_bbox{}.json'.format(fold), 'w'), indent=4)

print('Done')

