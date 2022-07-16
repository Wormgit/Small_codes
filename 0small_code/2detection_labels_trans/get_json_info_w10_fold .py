import sys,math
import os
import json
from xml.etree.ElementTree import ElementTree,Element

dir_name = os.path.abspath(os.path.dirname(__file__))
libs_path = os.path.join(dir_name, '..', 'libs')
sys.path.insert(0, libs_path)

json_path = '/home/io18230/Desktop/'
annotation_file = json_path + '10-fold-CV.json'
dataset = json.load(open(annotation_file, 'r'))

last = 'any'
difficult = 0

for a in range(0,10):
    t = ''+str(a)+''
    test = dataset[t]['test']

    for mm in range(len(test)):
        test[mm] = test[mm] + '\n'
    nnn="".join(test)

    with open('/home/io18230/Desktop/rr/'+'test_'+str(a)+'.txt','w+') as f2:
        f2.write(nnn)

    train = dataset[t]['train']
    for mm in range(len(train)):
        train[mm] = train[mm] + '\n'
    bb="".join(train)
    with open('/home/io18230/Desktop/rr/'+'train_'+str(a)+'.txt','w+') as f2:
        f2.write(bb)

    train = dataset[t]['valid']
    for mm in range(len(train)):
        train[mm] = train[mm] + '\n'
    bb="".join(train)
    with open('/home/io18230/Desktop/rr/'+'valid_'+str(a)+'.txt','w+') as f2:
        f2.write(bb)

print('Done')

