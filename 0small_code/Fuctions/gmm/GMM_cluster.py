#!/usr/bin/env python
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import numpy as np  # notes are in main_test
import os, cv2, time, glob, shutil, copy, random
import matplotlib.pyplot as plt
import sys, math, warnings
import argparse
import json

#tf
#import keras
#import tensorflow as tf

# pytorch
import torchvision
import torchvision.datasets as dset
import torchvision.utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#website
##最后的图比较有用
#https://towardsdatascience.com/understanding-anomaly-detection-in-python-using-gaussian-mixture-model-e26e5d06094b
#more about outlier detection -decomposition density
# https://scikit-lego.readthedocs.io/en/latest/outliers.html#BayesianGMMOutlierDetector-Demonstration

# 画图,测试 score_sample
#https://www.kaggle.com/albertmistu/detect-anomalies-using-gmm

def estimateGaussian(X):
    m = X.shape[0]
    #compute mean of X
    mu = (np.sum(X, axis=0)/m)
    # compute variance of X
    var = np.var(X, axis=0)
    print('mu:',mu, 'var:',var)
    return mu,var


def multivariateGaussian(X, mu, sigma):
    k = len(mu)
    sigma=np.diag(sigma)
    X = X - mu.T
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma)**0.5)) * np.exp(-0.5* np.sum(X @ np.linalg.pinv(sigma) * X,axis=1))
    return p  #probabilities range(0,1)


###the first example,  1 demention good result!

# X, y_true = make_blobs(n_samples=500, centers=1, cluster_std=0.60, random_state=5)
# X_append, y_true_append = make_blobs(n_samples=20,centers=1, cluster_std=5,random_state=5)
# X = np.vstack([X, X_append])
# X = X[:, ::-1]
#
# mu, sigma = estimateGaussian(X)
# p = multivariateGaussian(X, mu, sigma)
#
# plt.figure(figsize=(8,6))
# plt.scatter(X[:,0], X[:,1], marker="x", c=p, cmap='viridis')
# # Circling of anomalies
# epsilon = 0.01     #   0.02
# outliers = np.nonzero(p<epsilon)[0]
# plt.scatter(X[outliers,0], X[outliers,1], marker="o", facecolor="none", edgecolor="r", s=70)
# plt.colorbar()
# plt.show()





from sklearn.model_selection import train_test_split
X, y_true = make_blobs(n_samples=400, centers=5, cluster_std=0.60, random_state=1)
X_append, y_true_append = make_blobs(n_samples=50, centers=5, cluster_std=5,random_state=1)
X = np.vstack([X, X_append])
y_true = np.hstack([[0 for _ in y_true], [1 for _ in y_true_append]])
X = X[:, ::-1] # flip axes for better plotting
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.33, random_state=1, shuffle=True)
#plt.scatter(X_train[:,0],X_train[:,1],marker="x")
#plt.show()


######poor reaults.  That’s why now we will move to a Mixture of Gaussians for multiple clusters
# mu, sigma = estimateGaussian(X_train)
# p = multivariateGaussian(X_test, mu, sigma)

# plt.figure(figsize=(8,6))
# plt.scatter(X_test[:,0], X_test[:,1], marker="x", c=p, cmap='viridis')
# outliers = np.nonzero(p<0.001)[0]
# plt.scatter(X_test[outliers,0],X_test[outliers,1],marker="o",facecolor="none",edgecolor="r",s=70)
# plt.show()




# GMM GaussianMixture methods-----
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components = 5, covariance_type = 'full', random_state=0)
gm.fit(X_train)
k = -gm.score_samples(X_train)  # Compute the log-likelihood of each sample.
c=gm.predict_proba(X_train)     # Evaluate the components’ density for each sample.

plt.figure(figsize=(10,10)) # 仅仅好看
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.scatter(X_train[:,0], X_train[:,1], c=gm.predict_proba(X_train)[:,i], cmap='viridis', marker='x')
    plt.colorbar()
plt.show()

plt.figure(figsize=(4,3))
plt.scatter(X_train[:,0], X_train[:,1], c=-gm.score_samples(X_train), cmap='viridis', marker='x')
plt.title('Negative log-likelihood using GMM')
plt.colorbar()
plt.show()

# GMM GaussianMixture methods-----




# GMM GMMOutlierDetector methods----
X_train  = X
U  = X
from sklego.mixture import GMMOutlierDetector

ncomponents = 5
mod = GMMOutlierDetector(n_components=ncomponents, threshold=0.85).fit(X)

plt.figure(figsize=(6, 8))
plt.subplot(311)
plt.scatter(X[:, 0], X[:, 1], c=mod.score_samples(X), s=8)

plt.title(f"likelihood of points given mixture of {ncomponents} gaussians")
plt.colorbar()

plt.subplot(312)
plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8)
plt.title("outlier selection")
plt.colorbar()

plt.subplot(313)
U = np.random.uniform(-15, 15, (10000, 2))
plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8)
plt.title("outlier selection")
plt.colorbar()
plt.show()

plt.figure(figsize=(14, 3))
for i in range(1, 5):
    mod = GMMOutlierDetector(n_components=ncomponents, threshold=i/10, method="stddev").fit(X)
    plt.subplot(140 + i)
    plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8)  #-1 (outlier) or +1 (normal).
    plt.title(f"outlier sigma={i}")
plt.show()

U  = X
plt.figure(figsize=(14, 3))
for i in range(1, 5):
    mm = i/5
    mod = GMMOutlierDetector(n_components=ncomponents, threshold=mm, method="stddev").fit(X)
    plt.subplot(140 + i)
    plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8)  #-1 (outlier) or +1 (normal).
    plt.title(f"outlier sigma={mm}")
plt.show()
# GMM GMMOutlierDetector methods----




rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

from sklearn.ensemble import IsolationForest
data = np.r_[X_train,X_test,X_outliers]
iso = IsolationForest(contamination='auto')
iso.fit(data)
pred = iso.predict(data)



from sklearn.datasets import make_blobs
X,_ = make_blobs(n_features=4, centers=4, cluster_std=2.5, n_samples=1000)
plt.scatter(X[:,0], X[:,1],s=10)

### 不好用
# plt.figure(figsize=(8, 8))
# from sklearn.covariance import EllipticEnvelope
# ev = EllipticEnvelope(contamination=.1)
# ev.fit(X)
# cluster = ev.predict(X)
# plt.scatter(X[:,0], X[:,1],s=10,c=cluster)
# plt.show()

plt.figure(figsize=(8, 8))
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=25,contamination=.1)
pred = lof.fit_predict(data)
s = np.abs(lof.negative_outlier_factor_)
plt.scatter(data[:,0], data[:,1],s=s*10,c=pred)
plt.show()