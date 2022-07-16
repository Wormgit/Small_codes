# resize +padding
from PIL import Image, ImageOps
import random, argparse, os
import shutil

# python resize.py --h=50

parser = argparse.ArgumentParser()
parser.add_argument('--frame_file', default='/home/io18230/0Projects/keras-retinanet-master/path/ID/1627video_on_server/Crop1627', type=str)
parser.add_argument('--h', default=112, type=int)
parser.add_argument('--th', default=2.74, type=float)
args = parser.parse_args()

desired_size = args.h
save_path = args.frame_file+'_h'+str(desired_size)


def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if os.path.isdir(path):
            raise

def del_dirs(path):  # another format is in dncnn
    if os.path.isdir(path):
        shutil.rmtree(path)

def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    image.show()
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((th*h - w) // 2, 0))  # 将图像填充为中间图像，两侧为灰色的样式
    new_image.show()

    return new_image
max_ = 0
del_dirs(save_path)

for items in os.listdir(args.frame_file):
    Folder = args.frame_file + '/' + items
    makedirs(save_path + '/' + items)
    for file in sorted(os.listdir(Folder)):
        File = args.frame_file + '/' + items + '/' + file
        im = Image.open(File)
        width, height = im.size
        old_size = im.size
        ratio = float(desired_size) / min(old_size)

        new_w = ratio * max(old_size)
        if new_w > max_:
            max_ = new_w
        if new_w/desired_size < 1.8:  #delete 超过的吧,还没有写程序
            print(new_w,File, )
        newsize = (int(new_w), desired_size)

        im1 = im.resize(newsize)

        #padding
        background = Image.new('RGB', (int(args.th*desired_size), desired_size), (0, 0, 0))
        background.paste(im1, ( int(args.th*desired_size - int(new_w))//2 , 0//2))
        #background.show()
        background.save(save_path + '/' + items + '/' + file)

print(max_/desired_size, args.th) # if over 2.74
print('Done')