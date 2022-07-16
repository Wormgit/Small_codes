import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300
np.random.seed(0)

# 生成球形数据集，数据中心为(20,20)
# randn()返回300 * 2 的具有标准正态分布的数据，同时样本数据点增加20，以此(20, 20)为样本中心。
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

#  生成以(0,0)为中心的高斯数据，
# dot()返回的是两个数组的点积
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# np.vstack:按垂直方向(行顺序)堆叠数组构成一个新的数组
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# 训练高斯混合模型
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

# 产生测试数据
x = np.linspace(-20., 30.)  # linspace()指定的间隔内返回均匀间隔的数字。
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)  # np.meshgrid()—生成网格点坐标矩阵
XX = np.array([X.ravel(), Y.ravel()]).T  # ravel()：多维数组转换为一维数组，如果没有必要，不会产生源数据的副本

dd = clf.predict(XX)
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)
kk = clf.score(XX)
MMMM = clf.predict_proba(XX)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8, edgecolors=[1, 0, 0])  # scatter()画散点图
plt.scatter(XX[:, 0], XX[:, 1], .8)  # scatter()画散点图
plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()