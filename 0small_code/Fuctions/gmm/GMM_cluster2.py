#!/usr/bin/env python

import numpy as np  # notes are in main_test
import os, cv2, time, glob, shutil, copy, random
import matplotlib.pyplot as plt
import sys, math, warnings
import argparse
import json

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
def GMM_m(X_train, n_components=5):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gm.fit(X_train)
    # k = -gm.score_samples(X_train)  # Compute the log-likelihood of each sample.
    c = gm.predict_proba(X_train)  # Evaluate the components’ density for each sample.
    # predict_proba value
    plt.figure(figsize=(10, 10))
    for i in range(0, n_components):
        plt.subplot(321 + i)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=gm.predict_proba(X_train)[:, i], cmap='viridis', marker='x',
                    alpha=0.5)
    ##### score value
    plt.subplot(3, 2, 6)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=gm.score_samples(X_train), cmap='viridis', marker='x', alpha=0.5)
    plt.title('Negative log-likelihood GMM, score sample')
    plt.colorbar()
    plt.show()


# GMM GaussianMixture methods-----


# GMM GaussianMixture methods-----
def GMM_test(X_train, n_components = 5, th_each=0.1):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components = n_components, covariance_type = 'full', random_state=0)
    gm.fit(X_train)
    #k = -gm.score_samples(X_train)  # Compute the log-likelihood of each sample.
    c= gm.predict_proba(X_train)     # Evaluate the components’ density for each sample.
    #predict_proba value
    plt.figure(figsize=(10, 10))
    labels = gm.predict(X_train)
    for i in range(n_components):
        plt.subplot(321 + i)
        plt.scatter(X_train[:,0], X_train[:,1], c=gm.predict_proba(X_train)[:,i], cmap='viridis', marker='o',alpha=0.1)
        value = gm.predict_proba(X_train)[:,i]

        label_check = labels[np.argmax(value)]
        indices = np.where(labels ==label_check)[0]
        indices_value = value[indices]

        number_ = int(len(indices) * th_each)
        out = np.argsort(indices_value)
        kk = out[:number_] # the most th_each percent small data

        plt.scatter(X_train[indices[:], 0], X_train[indices[:], 1], c='b', marker='o', alpha=1)
        plt.scatter(X_train[indices[kk], 0], X_train[indices[kk], 1], c ='r', marker='o', alpha=1)


    ##### score value
    plt.subplot(3, 2, 6)
    plt.scatter(X_train[:,0], X_train[:,1], c=gm.score_samples(X_train), cmap='viridis', marker='x',alpha=0.5)
    plt.title('Negative log-likelihood GMM, score sample')
    plt.colorbar()
    plt.show()
# GMM GaussianMixture methods-----




# GMM GMMOutlierDetector methods----
def Gmm_outlier(X, ncomponents = 5, threshold= 0.85, Urange=[-15,10]):
    from sklego.mixture import GMMOutlierDetector
    mod = GMMOutlierDetector(n_components=ncomponents, threshold=threshold).fit(X)

    plt.figure(figsize=(5, 8))
    plt.subplot(311)
    plt.scatter(X[:, 0], X[:, 1], c=mod.score_samples(X), s=8, alpha=0.5)
    plt.title(f"likelihood of points given mixture of {ncomponents} gaussians")
    plt.colorbar()

    plt.subplot(312)
    plt.scatter(X[:, 0], X[:, 1], c=mod.predict(X), s=8, alpha=0.5)
    plt.title("outlier, predict, GMMOutlierDetector")
    plt.colorbar()

    plt.subplot(313)
    U = np.random.uniform(Urange[0], Urange[1], (10000, 2))
    plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8, alpha=0.5)
    plt.title("outlier mask purple")
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(8, 8))
    for i in range(1, 7, 2):
        mm = i / 5
        mod = GMMOutlierDetector(n_components=ncomponents, threshold=mm, method="stddev").fit(X)
        plt.subplot(320 + i)
        plt.scatter(U[:, 0], U[:, 1], c=mod.predict(U), s=8, alpha=0.5)  #-1 (outlier) or +1 (normal).
        plt.title(f"GMM outlier std sigma={mm}")

        plt.subplot(320 + i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=mod.predict(X), s=8, alpha=0.5)  #-1 (outlier) or +1 (normal).
    plt.show()

# GMM GMMOutlierDetector methods----



def IsolationForest(data):
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination='auto')
    iso.fit(data)
    pred = iso.predict(data)
    plt.scatter(data[:,0], data[:,1],s=8,c=pred)
    plt.title(f"IsolationForest")
    plt.show()

### 不好用
def EllipticEnvelope(X):
    plt.figure(figsize=(8, 8))
    from sklearn.covariance import EllipticEnvelope
    ev = EllipticEnvelope(contamination=.1)
    ev.fit(X)
    cluster = ev.predict(X)
    plt.scatter(X[:,0], X[:,1],s=10,c=cluster)
    plt.title(f"EllipticEnvelope")
    plt.show()


def LocalOutlierFactor(data):
    plt.figure(figsize=(8, 8))
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=25, contamination=.1)
    pred = lof.fit_predict(data)
    s = np.abs(lof.negative_outlier_factor_)
    plt.scatter(data[:,0], data[:,1],s=s*10,c=pred, alpha=0.5)
    plt.title(f"LocalOutlierFactor")
    plt.show()

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split

    ##dataset 1 --------- 2 clusters -- 2d
    X, y_true = make_blobs(n_samples=400, centers=5, cluster_std=0.60, random_state=1)           #main data
    X_append, y_true_append = make_blobs(n_samples=50, centers=5, cluster_std=5, random_state=1) #noise data
    X = np.vstack([X, X_append])                                                                 #combine them
    y_true = np.hstack([[0 for _ in y_true], [1 for _ in y_true_append]])
    X = X[:, ::-1]  # flip axes for better plotting
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.33, random_state=1, shuffle=True)

    LocalOutlierFactor(X_train)
    IsolationForest(X_train)
    GMM_test(X_train)#!!!!
    #GMM_test(X_train,th_each=0.05)  # !!!!
    GMM_m(X_train)
    Gmm_outlier(X_train)



    ##dataset 2 --------- 2 clusters -- 2d
    # rng = np.random.RandomState(42)
    # # Generate train data
    # X = 0.3 * rng.randn(100, 2)
    # X_train = np.r_[X + 2, X - 2]
    # X = 0.3 * rng.randn(20, 2)
    # X_test = np.r_[X + 2, X - 2]
    # # Generate some abnormal novel observations
    # X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    # data = np.r_[X_train, X_test, X_outliers]
    #
    # GMM_test(data, n_components = 2)
    # GMM_m(data, n_components = 2)
    # Gmm_outlier(data, ncomponents=2, threshold=0.85, Urange=[-4, 4])
    # LocalOutlierFactor(data)
    # IsolationForest(data)


