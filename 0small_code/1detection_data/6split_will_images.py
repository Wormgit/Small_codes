import json
import glob, os
import shutil


def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

type = ['valid','test','train']
for fold in range(0,10):
    for it in type:
        f = open('/home/io18230/0Projects/keras-retinanet-master/path/COWs/cross_val/split/'+it+'_'+str(fold)+'.txt') # test train
        path = '/home/io18230/Desktop/all'
        save_json_path = '/home/io18230/Desktop/rr/'+ str(fold) + '/' + it#val train
        makedirs(save_json_path)

        content=f.readlines()
        # for line in content:
        #     #image_id = line.strip('/home/will/work/1-RA/src/Detector/data/consolidated_augmented/')
        #     image_id = line.strip('\n')
        #     name, ext = os.path.splitext(image_id)
        #
        #     shutil.copy(os.path.join(path, image_id), os.path.join(save_json_path, image_id))
        #     #shutil.copy(os.path.join(path, name+'.txt'), os.path.join(save_txt_path, name+'.txt'))
        #     #print(name)
        #     shutil.copy(os.path.join(path, name+'.xml'), os.path.join(save_json_path, name+'.xml'))

        # for line in content:
        #     image_id = line.strip('\n')
        #     nam, ext = os.path.splitext(image_id)
        #     name = nam.zfill(6)
        #     shutil.copy(os.path.join(path, name+'.jpg'), os.path.join(save_json_path, name+'.jpg'))
        #     shutil.copy(os.path.join(path, name + '.xml'), os.path.join(save_json_path, name + '.xml'))

        for line in content:
            image_id = line.strip('\n')
            nam, ext = os.path.splitext(image_id)
            name = nam.zfill(6)
            if name == '00000':
                continue
            shutil.copy(os.path.join(path, name+'.jpg'), os.path.join(save_json_path, name+'.jpg'))
            shutil.copy(os.path.join(path, name + '.xml'), os.path.join(save_json_path, name + '.xml'))