# Core libraries
import os
import sys
sys.path.append(os.path.abspath("../"))
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill


# My libraries


class TrainingVisualiser(object):
	# Class constructor
	def __init__(self):
		pass

	@staticmethod
	def visualiseFromNPZFile(log_path, acc_path):
		print(f"Loading data from file: {log_path}")

		epoch_validation_gap = 1
		show_val_acc = 1
		Train_epoch = 61
		y_axi_loss = 10
		m_frontsize = 16

		if os.path.exists(root_dir+"/valLoss_All_mean.npz"):
			with np.load(root_dir+"/valLoss_All_mean.npz") as data:
				valLoss = data['valLoss_All_mean']

		Train_epoch= len(valLoss) # 可能val出的更慢一点
		print(f'Results of {Train_epoch} epochs')

		if os.path.exists(log_path):
			with np.load(log_path) as data:
				train_loss = data['losses_mean']
				train_steps = data['loss_steps']
				#mmval_acc = data['accuracies']/100    # original code we do not use it
				#mmval_acc = data['accuracies']/100    # original code we do not use it
				val_steps = data['accuracy_steps']

				train_size = int(len(train_steps) /Train_epoch)
				Train_step = train_steps[1] - train_steps[0]
				Epoch_trainloss = []
				for i in range(Train_epoch):
					fir = train_size * i
					las = train_size*(i+1) - 1
					ave = np.mean(train_loss[fir:las])
					Epoch_trainloss.append(ave)

				Epoch_train_steps = np.arange(Train_step * train_size, np.max(train_steps) + Train_step * train_size, Train_step * train_size)
				Epoch_train_steps = Epoch_train_steps[:len(Epoch_trainloss)]


			if val_steps.shape[0] == 0:
				step_size = round(float(np.max(train_steps)) /valLoss.shape[0])
				val_steps = np.arange(0, np.max(train_steps), step_size)
				val_steps = Epoch_train_steps

		if os.path.exists(acc_path):
			with np.load(acc_path) as bb:
				val_acc = bb['acc_test_folder']/100
				val_acc_ari = bb['ARI_test_folder']

			best_epoch_acc = np.argmax(val_acc)*epoch_validation_gap
			best_epoch_ari = np.argmax(val_acc_ari)*epoch_validation_gap
			print(f"Best accuracy Top 1 = {np.max(val_acc)}, @ Epoch{best_epoch_acc}")
			print(f"Best ARI            = {np.max(val_acc_ari)},@ use Epoch {best_epoch_ari} directly start from 0")
			for i in range (len(val_acc)):
				print(i,val_acc[i])
				if i >30:
					break

		#max_steps = max(np.max(val_steps), np.max(train_steps))
		max_steps = 63000 #np.max(train_steps)

		plt.rcParams['font.size'] = 14
		#plot accuracy
		fig, ax1 = plt.subplots()
		fig.set_figheight(6)
		fig.set_figwidth(8)
		colour1 = 'tab:blue'
		ax1.set_xlabel('Steps',fontsize =m_frontsize)
		ax1.set_ylabel('Accuracy or ARI', color=colour1,fontsize =m_frontsize)
		ax1.set_xlim((0, max_steps*1.35))
		ax1.set_ylim((0.,1.))

		if os.path.exists(acc_path):
			#ax1.plot(val_steps, val_acc, color=colour1, label='Test Acc', linestyle='-')
			step = val_steps
			if len(val_steps) > len(val_acc_ari):
				step = val_steps[:len(val_acc_ari)]
			if show_val_acc:
				ax1.plot(step, val_acc, color='tab:blue', label='val accuracy',linestyle='dashed')
			ax1.plot(step, val_acc_ari, color='tab:cyan', label='val ARI')  #tab:cyan orange  olive   tab:gray colour1

		# plot text
		#plt.scatter(val_steps[np.argmax(val_acc)], 1.02, markerfacecolor='none')
		if os.path.exists(acc_path):
			x_best = np.argmax(val_acc_ari)
			ax1.text(val_steps[x_best], np.max(val_acc_ari)+0.02, f'{round(np.max(val_acc_ari),2)}', fontsize=m_frontsize)
			#plt.plot(val_steps[x_best], np.max(val_acc_ari), 'o', color='tab:cyan', markersize=10, markerfacecolor='none')
			plt.scatter(val_steps[x_best], np.max(val_acc_ari),color='k')


		plt.legend(loc="upper right")
		#plot loss
		ax2 = ax1.twinx()
		colour2 = 'tab:red'
		ax2.set_ylabel('Reciprocal Triplet Loss', color='tab:orange',fontsize =m_frontsize)
		ax2.set_ylim((0., y_axi_loss))

		steps = val_steps
		if len(val_steps) > len(valLoss):
			steps = val_steps[:len(valLoss)]
		if os.path.exists(root_dir+"/valLoss_All_mean.npz"):
			l1=ax2.plot(steps, valLoss[:], color=colour2, label='val loss')


		ax2.plot(Epoch_train_steps, Epoch_trainloss, color='tab:orange', label='train loss')

		if os.path.exists(log_path):
			ax2.text(Epoch_train_steps[x_best], Epoch_trainloss[x_best] + 0.1,	 f'{round(Epoch_trainloss[x_best], 2)}',
					 fontsize=m_frontsize)
			plt.scatter(Epoch_train_steps[x_best], Epoch_trainloss[x_best], color='k')
		if os.path.exists(root_dir + "/valLoss_All_mean.npz"):
			ax2.text(val_steps[x_best], valLoss[x_best]-0.5, f'{round(valLoss[x_best], 2)}', fontsize=m_frontsize)
			plt.scatter(val_steps[x_best], valLoss[x_best], color='k')


		plt.legend(loc="lower right",borderaxespad=0.3)
		plt.tight_layout()
		#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
		plt.savefig(root_dir+'/train_id.pdf')
		plt.show()


# Entry method/unit testing method
if __name__ == '__main__':

	root_dir = '/home/io18230/Desktop/tmmmpid'#/aset'
	# valloss: valLoss_All_mean.npz needs to be in the directory.
	train_loss_path = os.path.join(root_dir, "logs.npz") # train loss
	acc_ARI_VAL = os.path.join(root_dir, "acc.npz")
	TrainingVisualiser.visualiseFromNPZFile(train_loss_path, acc_ARI_VAL)