#turn names like 2020-9-3 to 2020-09-03.
#check before using

import json, math
from xml.etree import ElementTree as et
import xml.dom.minidom as x
import glob, os, shutil

xml_path = '/home/io18230/Desktop/online_data/val'
destination = '/home/io18230/Desktop/2'

xmls = sorted(glob.glob(os.path.join(xml_path, '*.*')))  # *.xml

count = 0
fix_name =''

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

for xml in xmls:

    name = xml
    xml = xml.replace(xml_path+'/','')
    #hour = xml[position:position+2]

    position = 14             # fix minute first
    minute = xml[position:position+2]
    judge_m = minute[1]
    if is_number(judge_m):
        fix_minute = xml
    else:
        fix_minute = xml[:position] + '0' + xml[position: ]

    position = position + 3    # fix second
    second = fix_minute[position:position+2]
    judge_s = second[1]
    if is_number(judge_s):
        fix_name = fix_minute
    else:
        fix_name = fix_minute[:position] + '0' + fix_minute[position: ]

    shutil.copy(os.path.join(xml_path, name), os.path.join(destination, fix_name))


