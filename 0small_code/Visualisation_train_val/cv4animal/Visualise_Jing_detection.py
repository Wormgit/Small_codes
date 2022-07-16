
import os
import sys
sys.path.append(os.path.abspath("../"))
import numpy as np
from numpy import mean
import json
import matplotlib.pyplot as plt

json_path = '/home/io18230/Desktop/cv4animal/val/'  #

bias = 1 #if find epoch number, set it to zero.
m_frontsize = 18
txt_frontsize = 21
line = 2.5
marksize=80

first = 1
val_accf_stp= [1,100]
val_acc_1st = [0,0.028436]

tra_accf_stp= [1,100]
tra_acc_1st = [0,0.03] #####

valLoss_stp = [1,100]
valLoss_1st = [3.3457274436, 2.702]
traLoss_stp = [1,100]
traLoss_1st = [4.030566692352295,3.33] ####


def visualiseTrainingGraph(iou, epochs, val_loss, val_acc, train_acc = None,  batch= None , train_loss=None):

        err = epochs - len(train_acc)
        if err != 0:
            print (f'did not log the last {err}')

        plt.rcParams['font.size'] = 15

        with plt.rc_context(
                {'ytick.color': 'tab:blue'}):
            fig, ax1 = plt.subplots()
            fig.set_figheight(6)
            fig.set_figwidth(8)

            colour1 = 'tab:blue'
            ax1.set_xlabel('Steps', fontsize=m_frontsize)
            ax1.set_ylabel('Accuracy', color=colour1, fontsize=m_frontsize)
            #ax1.set_title('Accuracy and Loss @ IOU = {}'.format(iou))
            ax1.set_xlim([-700, np.max(epochs)* step_per_epoch*1.4]) #
            ax1.set_ylim((0, 1.))


            # if train_acc:
            step = range((0 + bias) * step_per_epoch, (epochs + bias -err) * step_per_epoch, step_per_epoch)
            if first:
                step =tra_accf_stp + list(step)
                train_acc = tra_acc_1st + train_acc
                val_acc = val_acc_1st + val_acc
            ax1.plot(step, train_acc, color='tab:cyan',label= 'train acc', linewidth= line)
            ax1.plot(step, val_acc[:-err], color='tab:blue', label='val acc',linewidth= line)
            plt.legend(loc="upper right", borderaxespad=0.2)

        # plot text
        x_best = val_acc.index(max(val_acc[:-err]))
        ax1.text(step[x_best], max(val_acc[:-err]) - 0.11, f'{round(max(val_acc), 2)}', fontsize=m_frontsize)
        plt.scatter(step[x_best], max(val_acc[:-err]), color='k',s = marksize)
        ax1.text(step[x_best], max(train_acc) - 0.06, f'{round(max(train_acc), 2)}', fontsize=m_frontsize)
        plt.scatter(step[x_best], max(train_acc), color='k',s = marksize)


        #plot loss
        with plt.rc_context(
                {'ytick.color': 'tab:orange'}):
            ax2 = ax1.twinx()
            colour2 = 'tab:red'
            ax2.set_ylabel('Loss', color='tab:orange',fontsize=m_frontsize)
            ax2.set_ylim((0., 4.))

            if first:
                val_loss = valLoss_1st + val_loss
                train_loss =traLoss_1st +train_loss
            ax2.plot(step, val_loss[:-err], color=colour2,label= 'val loss', linewidth= line)


            if batch:
                step2 = range((0 + bias) , (epochs -err) * step_per_epoch+1)
                ax2.plot(step2, batch[:-err], color='orange', label='train_loss')
                ax2.text(step[x_best], val_loss[x_best] - 0.01, f'{round(val_loss[x_best], 2)}', fontsize=m_frontsize)
                plt.scatter(step[x_best], val_loss[x_best], color='k',s = marksize)
            else:
                ax2.plot(step, train_loss[:-err], color='tab:orange', label='train loss' ,linewidth= line)
                ax2.text(step[x_best], train_loss[x_best] + 0.1, f'{round(train_loss[x_best], 2)}', fontsize=m_frontsize)
                plt.scatter(step[x_best], train_loss[x_best], color='k',s = marksize)
            plt.legend(loc="lower right", borderaxespad=0.2)

        ax2.text(step[x_best], val_loss[x_best] - 0.23, f'{round(val_loss[x_best], 2)}', fontsize=m_frontsize)
        plt.scatter(step[x_best], val_loss[x_best], color='k',s = marksize)

        # right side text:
        # GAP = 0.08
        # start = 0.95
        # x_pos = 1.5
        # ax1.text(x_line*x_pos, 0.95, f'Test Parameters', fontsize=txt_frontsize)
        # ax1.text(x_line * x_pos, 0.95 - GAP*1,  f'mAP = {m_frontsize}', fontsize=txt_frontsize)
        # ax1.text(x_line * x_pos, 0.95 - GAP*2,  f'IOU:                {iou}', fontsize=txt_frontsize)
        # ax1.text(x_line * x_pos, 0.95 - GAP*3,  f'NMS th:          {0.2}', fontsize=txt_frontsize)
        # ax1.text(x_line * x_pos, 0.95 - GAP * 4,f'Confidence th:{0.3}', fontsize=txt_frontsize)
        print (step)
        plt.tight_layout()
        plt.savefig(json_path + str(number)+ '_'+str(iou)[-1], bbox_inches='tight', pad_inches=0.2)
        plt.savefig(json_path + str(number) + '_' + str(iou)[-1] + '.pdf', bbox_inches='tight', pad_inches=0.2)
        plt.show()

for number in range(1):
    json_ = json_path + str(number) + '.json'
    with open(json_,'r') as load_f:
        his = json.load(load_f)
        epochs = len(his['loss'])
        batch_loss =[]
        batch = []
        for id, item in enumerate(his['batch']):
            batch_loss.append([float(x) for x in item])
        for item in batch_loss:
            step_per_epoch = len(item)
            for i in range(len(item)):
                ave = mean(item[:i+1])
                batch.append(ave)

        # # IOU = 0.5 show training loss per Batch? yes: batch=batch  NO,  batch=None
        # visualiseTrainingGraph(0.5, epochs, his['val_loss'], his['val_acc'], train_acc=his['acc'], batch=None, train_loss=his['loss'])
        # # IOU = 0.7 show training loss per Batch? yes: batch=batch  NO,  batch=None
        # print('xx  epoch(match the trained strat at 1), valuse')
        # print (number, his['val_acc'].index(max(his['val_acc']))+1, max(his['val_acc']) )

        visualiseTrainingGraph(0.7, epochs, his['val_loss'], his['val_acc7'], train_acc=his['acc7'], batch=None, train_loss=his['loss']) #batch=batch
        # for i,j in enumerate(his['val_acc7']):
        #     print(i, j)
        print('xx  use this epoch(match the trained strat at 1), valuse')
        print(number, his['val_acc7'].index(max(his['val_acc7']))+1, max(his['val_acc7']))

        # stable epoch     best epoch(0.5)   best acc      best epoch(0.7)   best acc
        # 0    55              11 0.9861560685730499         37 0.9640336571823217
        # 1    67              32 0.9838767839492774         21 0.9540953064324181
        # 2    70              24 0.9963817966038239         51 0.9683993759076974
        # 3    81              40 0.9922839609781698         67 0.9657550942728269
        # 4    90              20 0.9885722692641016         63 0.9539748
        # 5    64              54 0.9947461                  54 0.96962
        # 6    68              18 0.994941                   24 0.969923
        # 7    114             18 0.9855300                  58 0.947095
        # 8    66              36 0.99153                    18 0.956425
        # 9    54              38 0.9875                     41 0.9612067