import sys,math
import os
import json
from xml.etree.ElementTree import ElementTree,Element

dir_name = os.path.abspath(os.path.dirname(__file__))
libs_path = os.path.join(dir_name, '..', 'libs')
sys.path.insert(0, libs_path)
from pascal_voc_io import PascalVocWriter

#json_path = '/home/io18230/Desktop/online_data/block_272/'
json_path = '/home/io18230/Desktop/00/'
annotation_file = json_path + 'results_bbox_square.json'
#annotation_file = json_path + '00.json'
dataset = json.load(open(annotation_file, 'r'))

def read_xml(in_path):
    tree = ElementTree()
    tree.parse(in_path)
    return tree

def write_xml(tree, out_path):
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

def create_node(tag, content,node):
    element = Element(tag)
    element.text = content
    node.append(element)


last = 'any'
difficult = 0
for a in dataset:
    name = a['file_name'].strip('.jpg')
    s = a['score']
    box = a['bbox']

    xe = box[0] + box[2]/ 2
    ye = box[1] + box[3]/ 2
    we = box[3]

    #endx = int(xe + we * 0.3 * (math.cos(box[4])))
    #endy = int(ye + we * 0.3 * (math.sin(box[4])))

    if name != last:  #The first one
        writer = PascalVocWriter(foldername='tests', filename=name, imgSize=(a['height'], a['width'], 3), localImgPath='tests/test.png')
        #writer.addRotatedBndBox(xe,ye, box[2], box[3], box[4] , 'cow', difficult)
        writer.addRotatedBndBox(xe, ye, box[2], box[3],0, 'cow', difficult)
        #writer.addRotatedBndBox(endx, endy,1, 1, 0, 'head' , difficult)
        writer.save('1/'+name+'.xml')

        tree = read_xml('1/' + name + '.xml')
        root = tree.getroot()  # 获取xml文件的根节点

        #add confidence to the first one
        elem02 = root.find("object")
        create_node('score', str(s), elem02)
        write_xml(tree, '1/' + name + '.xml')

    else:
        tree = read_xml('1/'+name+'.xml')
        root = tree.getroot()  # 获取xml文件的根节点

        object = Element('object')
        create_node('type', 'robndbox', object)
        create_node('name', 'cow', object)
        create_node('pose', 'Unspecified', object)
        create_node('truncated', '0', object)
        create_node('difficult', '0', object)


        element = Element('robndbox')
        object.append(element)

        create_node('cx',str(xe), element)
        create_node('cy',str(ye), element)
        create_node('w', str(box[2]), element)
        create_node('h', str(box[3]), element)
        #create_node('angle', str(box[4]), element)
        create_node('score', str(s), object)
        root.append(object)

        # head = Element('object')
        # create_node('type', 'robndbox', head)
        # create_node('name', 'head', head)
        # create_node('pose', 'Unspecified', head)
        # create_node('truncated', '0', head)
        # create_node('difficult', '0', head)
        #
        # element2 = Element('robndbox')
        # head.append(element2)
        #
        # create_node('cx',str(endx), element2)
        # create_node('cy',str(endy), element2)
        # create_node('w', str(1), element2)
        # create_node('h', str(1), element2)
        # create_node('angle', str(0), element2)
        # root.append(head)

        write_xml(tree, '1/'+name+'.xml')
    last = name

print('Done')

