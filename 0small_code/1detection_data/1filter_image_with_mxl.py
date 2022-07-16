#move jpg and xml to another file(only the one with xml file).
#change the original name to '0123.jpg'
#input: start_file, end_file, path, (images.jpg and images.xml)

import glob, os
import shutil
from collections import Counter

path = '/home/io18230/0Projects/keras-retinanet-master/path/Detection_rota/Rotate_inbarn/label/test'
destination = '/home/io18230/0Projects/keras-retinanet-master/path/Detection_rota/Rotate_inbarn/Test'
img = '/home/io18230/0Projects/keras-retinanet-master/path/Detection_rota/Rotate_inbarn/download'
count = 8271

'''
source are image like 2020-03-12

'''


for x in sorted(os.listdir(path)):
    for item in sorted(os.listdir(os.path.join(path, x))):
        print(x, item)
        name, ext = os.path.splitext(item)
        c = "%05d" % count
        count = count + 1
        shutil.copy(os.path.join(path, x, item), os.path.join(destination, c+'.xml'))
        shutil.copy(os.path.join(img, x, name+'.jpg'), os.path.join(destination, c+'.jpg'))




#filter image with xml and give count name
# for x in sorted(os.listdir(path)):
    # if  x.endswith('.xml'):
    #     name, ext = os.path.splitext(x)
    #     if os.path.exists(os.path.join(path, name)+'.jpg'):
    #         c = "%06d" % count
    #         count = count +1
    #         shutil.copy(os.path.join(path, x), os.path.join(destination, c+'.xml'))
    #         shutil.copy(os.path.join(path, name)+'.jpg', os.path.join(destination, c+'.jpg'))
    #     else:
    #         print (error)

#do not filter image. give count name only
# for x in sorted(os.listdir(path)):
#     if  x.endswith('.jpg'):
#         name, ext = os.path.splitext(x)
#         count = count + 1
#         c = "%06d" % count
#         shutil.copy(os.path.join(path, x), os.path.join(destination,  c+'.jpg'))
#
#         if os.path.exists(os.path.join(path, name)+'.xml'):
#             shutil.copy(os.path.join(path, name)+'.xml', os.path.join(destination, c+'.xml'))


#filter image with xml and skip count name without xml(name number != count).
# for x in sorted(os.listdir(path)):
#     if  x.endswith('.jpg'):
#         name, ext = os.path.splitext(x)
#         count = count + 1
#         c = "%06d" % count
#
#         if os.path.exists(os.path.join(path, name)+'.xml'):
#             shutil.copy(os.path.join(path, x), os.path.join(destination, c + '.jpg'))
#             shutil.copy(os.path.join(path, name)+'.xml', os.path.join(destination, c+'.xml'))


# filter image without xml and skip count name with xml(name number != count).
# for x in sorted(os.listdir(path)):
#     if  x.endswith('.jpg'):
#         name, ext = os.path.splitext(x)
#         count = count + 1
#         c = "%06d" % count
#         if os.path.exists(os.path.join(path, name) + '.xml'):
#             pass
#         else:
#             shutil.copy(os.path.join(path, x), os.path.join(destination, c + '.jpg'))

#b = dict(Counter(list))
#print('\nmissing files are')
#print ({key:value for key,value in b.items()if value == 1})

# anot = '/home/io18230/0Projects/keras-retinanet-master/path/Rotate_inbarn/all_Rotate_inbarn_rgb'
#
# for x in sorted(os.listdir(path)):
#     if  x.endswith('.jpg'):
#         name, ext = os.path.splitext(x)
#         shutil.copy(os.path.join(anot, x), os.path.join(destination, x))
#         shutil.copy(os.path.join(anot, name + '.xml'), os.path.join(destination, name + '.xml'))