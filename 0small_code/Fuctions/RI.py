# import numpy as np
# import matplotlib.pyplot as plt
#
# label = [0,0,1,0,1,0,0,1]
# pred = [0,0,0,0,0,1,1,1]
# bins = np.array([0,0.5,1]) # 表示x,y轴每个 bin 的坐标范围
# tn, fp, fn, tp = plt.hist2d(label, pred, bins=bins, cmap='Blues')[0].flatten()
def RandIndedx(x, y):
    l = len(x)
    agrees = 0
    for i in range(l-1):
        for j in range(i+1, l):
            x_status = int(x[i] == x[j])
            y_status = int(y[i] == y[j])
            agrees += int(x_status == y_status)
    return agrees * 2 / (l * (l - 1))
#
#
#
from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

# 基本用法 # 具有对称性 # 与标签名无关
score = metrics.adjusted_rand_score(labels_true, labels_pred)

print(score)
print(RandIndedx(labels_true,labels_pred))
print('\n')

# 接近 1 最好
labels_pred = [0, 0, 1, 1, 0, 0]
#labels_pred = labels_true[:]
score = metrics.adjusted_rand_score(labels_true, labels_pred)

print(score)
print(RandIndedx(labels_true,labels_pred))

print('\n')
labels_pred = [0, 0, 1, 1, 1, 1]
score = metrics.adjusted_rand_score(labels_true, labels_pred)
print(score)
print(RandIndedx(labels_true,labels_pred))

# 独立标签结果为负或者接近 0
labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
score = metrics.adjusted_rand_score(labels_true, labels_pred)
print(score)
print(RandIndedx(labels_true,labels_pred))



