
import os
import sys
sys.path.append(os.path.abspath("../"))
import numpy as np
from numpy import mean
import json
import matplotlib.pyplot as plt

mode = 18
json_path = '/home/io18230/Desktop/'  #  /home/io18230/0Projects/metric/SimCLR/
json_number = 214
json_ = json_path + 'log_loss'+str(mode)+'_'+str(json_number)+'.json'
bias = 1


def visualiseTrainingGraph(name, epochs, val_loss, val_acc, train_acc = None,  batch= None , train_loss=None):

        with plt.rc_context(
                {'ytick.color': 'tab:blue'}):
            fig, ax1 = plt.subplots()
            colour1 = 'tab:blue'
            ax1.set_xlabel('Steps  ({} steps/epoch)'.format(step_per_epoch))
            ax1.set_ylabel('Accuracy', color=colour1)
            ax1.set_title('Accuracy and Loss')
            ax1.set_xlim([0, np.max(epochs)* step_per_epoch*1.35]) # 1
            ax1.set_ylim((0.5, 1.))
            if val_acc:
                ax1.plot(range((0+bias)*step_per_epoch,(epochs+bias)*step_per_epoch,step_per_epoch), val_acc, color='tab:blue',label= 'val_acc')
            if train_acc:
                ax1.plot(range((0+bias)*step_per_epoch,(epochs+bias)*step_per_epoch,step_per_epoch), train_acc, color='tab:cyan',label= 'train_acc')
            plt.legend(loc="lower right")

        #plot loss
        with plt.rc_context(
                {'ytick.color': 'tab:orange'}):
            ax2 = ax1.twinx()
            colour2 = 'tab:red'
            ax2.set_ylabel('Loss', color='tab:orange')
            #ax2.set_ylim((0., 3.))

            if batch:
                ax2.plot(range((0 + bias) , (epochs ) * step_per_epoch+1), batch, color='orange', label='train_loss')
            else:
                ax2.plot(range((0 + bias) * step_per_epoch, (epochs + bias) * step_per_epoch, step_per_epoch),
                         train_loss, color='tab:orange', label='train_loss')
            if val_loss:
                ax2.plot(range((0 + bias) * step_per_epoch, (epochs + bias) * step_per_epoch, step_per_epoch), val_loss,
                     color=colour2, label='val_loss')
            plt.legend(loc="upper right")
        plt.tight_layout()
        #plt.show()
        plt.savefig(json_path + 'loss_' + name +'_' +str(json_number), bbox_inches='tight', pad_inches=0.2)


with open(json_,'r') as load_f:
    his = json.load(load_f)
    epochs = len(his['batch'])
    batch_loss =[]
    batch = []
    for id, item in enumerate(his['batch']):
        batch_loss.append([float(x) for x in item])
        
    for item in batch_loss: # average the used ones
        step_per_epoch = len(item)
        for i in range(len(item)):
            ave = mean(item[:i+1])
            batch.append(ave)

    his['loss'] = []
    for item in batch_loss:
        his['loss'].append(np.mean(item))


    visualiseTrainingGraph('every_epoch', epochs, his['val_loss'], his['val_acc'], train_acc=his['acc'], batch=None, train_loss=his['loss'])
    # show training loss per Batch? yes: batch=batch  NO,  batch=None
    #print (his['val_acc'].index(max(his['val_acc'])), max(his['val_acc']) )
    # for i,j in enumerate(his['val_acc']):
    #     print(i, j)
    #visualiseTrainingGraph('every_step',epochs, his['val_loss'], his['val_acc'], train_acc=his['acc'], batch=batch, train_loss=his['loss'])

print('Done')

