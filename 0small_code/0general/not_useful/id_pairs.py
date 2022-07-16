import random
import argparse
import pandas as pd
import os, copy
import csv

# output file                 # 1 is easy, 0 is difficult (similar)
# anchor  positive  negative  (similarity of anchor & p)  (similarity of anchor & n)
from shutil import copyfile
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--frame_file', default='/home/io18230/Desktop/dat', type=str) # from1627 from4  F1627simple
parser.add_argument('--negative_track', default=0.5, type=float) # Negative track : random ratio 2:1 0.3
parser.add_argument('--positive_track', default=4*20, type=int) # positive for each track 5 200
parser.add_argument('--agument_n', default=5, type=int) # positive for each track
args = parser.parse_args()
save_path = args.frame_file + '_csv_pari/'
agument_n = args.agument_n


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def difficulty(pair1,pair2):  # 1 is easy, 0 is difficult (similar)
    m1 = pair1[13]
    m2 = pair2[13]
    if m1 == m2:
        return '0' #difficult and similar based on histogram 0
    elif m1=='T' and m2 =='B':
        return '0'
    elif m2 == 'T' and m1 == 'B':
        return '0'
    else:     # T/B+ W
        return '1' # easy






def positive_track(csv_path, items):
    # positive pair: different images, same video
    # print(items)
    C_positive = 0
    count_n_r = 0
    C_neg = 0
    #trouble = 0
    csvFile = csv.reader(open(csv_path, 'r'), delimiter=',')
    all = sorted(os.listdir(args.frame_file))

    reader = list(csvFile)
    del reader[0]
    for i in range(len(reader)): # delete blanks from csv format and track points with out valid box
        while '' in reader[i]:
            reader[i].remove('')
    count = len(reader)-1
    while True:
        if len(reader[count]) < 2:
            del reader[count]
        count -= 1
        if count < 0 :
            break

    for i in range(len(reader)):
        count_pair = 1
        ran_neg = -1 # negative track
        th_only_rand_neg = args.negative_track
        #lenght of track rather than valid cattle. so the real one should be less than it.


        item_n = 0
        xmls = sorted(os.listdir(args.frame_file + '/' + items))
        same_list = []
        dest = args.frame_file + 'split/' + items + '/'+str(i)
        makedirs(dest)
        for tmm in range(1,len(reader[i])): # random 2 items search for 2 different items else go on   #positive pairs
            # if len(reader[i])==2:
            #     break
            if tmm == 1:
                ig = eval(reader[i][tmm])
                xc = ig[0]
                yc = ig[1]

                while 1:
                    xml = xmls[item_n]
                    x, y = int(xml[17:21]), int(xml[22:26])
                    if xc == x and yc == y:

                        for kkk in range(0,5):
                            same_list.append(xmls[item_n+kkk])
                        item_n = 0
                        break
                    item_n+=5
            else:
                ig = eval(reader[i][tmm])
                xc = ig[0]
                yc = ig[1]
                    #ig_number = ig[-1]
                #while item_n < len(xmls):
                while 1:
                    xml = xmls[item_n]
                    x, y = int(xml[17:21]), int(xml[22:26])
                    if xc == x and yc == y:

                        for kkk in range(0, 5):
                            same_list.append(xmls[item_n + kkk])
                        item_n = 0
                        break
                    item_n += 5
        print(same_list)
        for it in same_list:
            shutil.copy(os.path.join(args.frame_file+'/'+ items, it), os.path.join(dest, it))

        same_list = []


    print(1)

    #         if len(reader) > 1:
    #             while True: #neg
    #                 ran_neg = random.randint(0, len(reader) - 1)
    #                 if len(reader[ran_neg])<2:
    #                     continue
    #                 if ran_neg != i:  # different row of csv
    #                     img = random.randint(1, len(reader[ran_neg]) - 1)
    #                     ran_neg_ = eval(reader[ran_neg][img])
    #                     xcn = ran_neg_[0]
    #                     ycn = ran_neg_[1]
    #                     break
    #         else:
    #             th_only_rand_neg = -1
    #
    #
    #
    #             xmls = sorted(os.listdir(args.frame_file + '/' + items))
    #             while 1: # till find the
    #
    #
    #                     image_name = xml[position:position + 6]
    #                     if search_1:
    #                         # if int(image_name) > ig_number:
    #                         #     trouble = 1
    #                         #     break
    #                         if int(image_name) == ig_number:# the first one
    #                             x, y = int(xml[17:21]), int(xml[22:26])
    #                             if xc == x and yc == y:
    #                                 pair_n1 = item_n + random.randint(0, 4)  #random choose a agumneted angle
    #                                 pair1 = xmls[pair_n1]
    #                                 search_1 = 0
    #                             item_n += agument_n
    #                             continue
    #
    #                     if int(image_name) == ig_number2:
    #                         x2, y2 = int(xml[17:21]), int(xml[22:26])
    #                         if xc2 == x2 and yc2 == y2:
    #                             pair_n2 = item_n + random.randint(0, 4)
    #                             pair2 = xmls[pair_n2]
    #
    #                             if random.uniform(0, 1) <= th_only_rand_neg:
    #                                 while 1:  # till find the negative track
    #                                     xml_n = xmls[neg_n]
    #                                     if len(xml_n) > 20:
    #                                         image_name_n = xml_n[position:position + 6]
    #                                         if int(image_name_n) == ran_neg_[-1]:
    #                                             xn, yn = int(xml_n[17:21]), int(xml_n[22:26])
    #                                             if xcn == xn and ycn == yn:
    #                                                 pair_n3 = neg_n + random.randint(0, 4)
    #                                                 pair3 = xmls[pair_n3]
    #                                                 C_neg += 1
    #                                                 pair_list.append([items + '/' + pair1, items + '/' + pair2, items + '/' + pair3,  difficulty(pair1,pair2), difficulty(pair1,pair3)])
    #                                                 break
    #                                         neg_n += agument_n
    #                             else:
    #                                 while 1:
    #                                     sample = random.sample(all, 1)
    #                                     # if sample[0] == items:
    #                                     #     print(sample)
    #                                     if sample[0] != items:
    #                                         all2 = sorted(os.listdir(args.frame_file + '/' + sample[0]))
    #                                         pair3 = "".join(random.sample(all2, 1))
    #                                         #if difficulty(pair1, pair3) == '1':
    #                                         pair_list.append(
    #                                             [items + '/' + pair1, items + '/' + pair2, sample[0] + '/' + pair3,
    #                                              difficulty(pair1, pair2), difficulty(pair1, pair3)])
    #                                         count_n_r += 1
    #                                         break
    #
    #                             #print(pair1, pair2)
    #                             count_pair += 1
    #                             break_22 = 1
    #                             C_positive=C_positive+1
    #                             #print(pair_n2)
    #                     item_n += agument_n
    #                     #print(item_n, ig_number, ig_number2, xml)
    #                 if break_22:
    #                     break_22 = 0
    #                     break
    #
    # return C_positive, C_neg ,count_n_r

makedirs(save_path)
position = 2

# load info
c_image = []
c_folder= []
pair_list= []
count_p=0
count_n=0
count_n_r = 0
for items in os.listdir(args.frame_file):
    File = args.frame_file + '/' + items
    csv_path = save_path+'csv/'+items+'/same.csv'
    positive_track(csv_path, items)


print('Done')
