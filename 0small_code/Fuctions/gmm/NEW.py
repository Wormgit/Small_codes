# -*- coding: utf-8 -*-

from sklearn.mixture import GaussianMixture
import os, torch
import numpy as np
import cv2
import sys
#用高斯分布模型（GMM）对输入进行概率预测，输出概率列表，与给定阈值进行比较，最后输出符合条件的下标

a = torch.randn(5,1)
print(a)
gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, reg_covar=5e-4)
gmm.fit(a)

prob = gmm.predict_proba(a)
prob_list = gmm._estimate_weighted_log_prob(a)
print(prob)
print(gmm.means_)
means = gmm.means_.argmin()
print(means)

prob = prob[:, gmm.means_.argmin()]
print(prob)
pred = prob > 0.5
print(pred)
print(pred.nonzero())
pred_idx = pred.nonzero()[0]
print(pred_idx)


#If `b` is given then ``np.log(np.sum(b*np.exp(a)))   logsumexp


