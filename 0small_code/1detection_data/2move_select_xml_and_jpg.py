#move jpg and xml to another file. count the missing xml file.
#sutible for
#input: start_file, end_file, path, path

import glob, os
import shutil
from collections import Counter

path = '/home/io18230/0Projects/keras-retinanet-master/keras_retinanet/path/to/COWs/images/train'
destination = '/home/io18230/Desktop/train'

start_file_number = 2287
end_file = 2347
list = []
last = start_file_number-1
for x in sorted(os.listdir(path)):
    if x.endswith('.jpg') or x.endswith('.xml'):
        name, ext = os.path.splitext(x)
        i_n = int(name)

        if i_n >= start_file_number and i_n <= end_file:
             list.append(i_n)
             shutil.copy(os.path.join(path, x), os.path.join(destination, x))

b = dict(Counter(list))
print('\nmissing files are')
print ({key:value for key,value in b.items()if value == 1})