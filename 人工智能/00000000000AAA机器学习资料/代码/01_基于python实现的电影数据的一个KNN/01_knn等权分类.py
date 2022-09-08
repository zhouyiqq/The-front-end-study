# -*- coding:utf-8 -*-
"""
# datetime: 14:34
# software: PyCharm
"""
'''
等权分类
'''
import numpy as np
import pandas as pd

##初始化训练数据
T = [[3, 104, -1],
     [2, 100, -1],
     [1, 81, -1],
     [101, 10, 1],
     [99, 5, 1],
     [98, 2, 1]
     ]
##预测数据
x_test = [18, 90]
##邻居
K = 5
###列表 [[dis1，标签1]，[dis2，标签2].。。。。。。]
listdistance = []
##循环每一个数据，计算他的dis
for t in T:  ## t是每条电影的数据
    dis = np.sum((np.array(t[:-1]) - np.array(x_test)) ** 2) ** 0.5
    listdistance.append([dis, t[-1]])
# print(listdistance)
##按照dis进行排序
listdistance.sort()

# print(listdistance)
##选取K个邻居放入投票箱
# print(listdistance[:K])
arr = np.array(listdistance[:K])[:, -1]
print(arr)
##统计投票
a = pd.Series(arr).value_counts()
print(a)
pre = a.idxmax()
print(pre)
