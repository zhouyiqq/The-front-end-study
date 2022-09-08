# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


##信息熵
def entropy(p):
    return np.sum([-i * np.log2(i) for i in p if i != 0])


##信息熵
def H(p):
    return entropy(p)


##基尼系数
def gini(p):
    return 1 - np.sum([i * i for i in p])


##错误率
def error(p):
    return 1 - np.max(p)


# # 不纯度
P0 = np.linspace(0.0, 1.0, 1000)  # 这个表示在0到1之间生成1000个p值(另外的就是1-p)
P1 = 1 - P0
y1 = [H([p0, 1 - p0]) * 0.5 for p0 in P0]  ##熵之半
y2 = [gini([p0, 1 - p0]) for p0 in P0]
y3 = [error([p0, 1 - p0]) for p0 in P0]
plt.plot(P0, y1, label=u'熵之半')
plt.plot(P0, y2, label=u'gini')
plt.plot(P0, y3, label=u'error')
plt.legend(loc='upper left')
plt.xlabel(u'p', fontsize=18)
plt.ylabel(u'不纯度', fontsize=18)
plt.title(u'不纯度指标', fontsize=20)
plt.show()  # 绘制图像

# # 熵和熵之半
y = [H([p0, 1 - p0]) for p0 in P0]
y1 = [H([p0, 1 - p0]) * 0.5 for p0 in P0]
plt.plot(P0, y, label=u'H(X)')
plt.plot(P0, y1, label=u'熵之半')
plt.legend(loc='upper left')
plt.xlabel(u'p', fontsize=18)
plt.ylabel(u'熵', fontsize=18)
plt.title(u'熵和熵之半', fontsize=20)
plt.show()  # 绘制图像
