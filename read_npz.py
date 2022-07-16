# -*- coding: utf-8 -*-


import os
import numpy as np
import cv2
import sys

embdingpath = '/home/io18230/Desktop/'
#embdingpath = '/home/io18230/Desktop/May/modi'

#embdingpath ='/home/io18230/Desktop/output/1-1_12-30/1Will'

#########################
# embedig_name='logs.npz'
# W_embeddings = np.load(os.path.join(embdingpath,embedig_name))
#
# loss = W_embeddings['losses_mean']
#
# W_embeddings2 = np.load(os.path.join(embdingpath,'valLoss_All_mean.npz'))
# vlaloss = W_embeddings2['valLoss_All_mean']

######################
#embedig_name='logs.npz'

embedig_name='folder_embeddings.npz'

W_embeddings = np.load(os.path.join(embdingpath,embedig_name))



# losses_mean = W_embeddings['losses_mean']
# loss_steps= W_embeddings['loss_steps']
# for item in losses_mean:
#     print(round(item,2))



e_path = W_embeddings['path']
# for item in e_path:
#     print(item)

l = W_embeddings['labels_knn']
M = W_embeddings['labels_correct_knn']
b= W_embeddings['labels_folder']
mmm = np.max(b)
e= W_embeddings['embeddings']
m = 1

