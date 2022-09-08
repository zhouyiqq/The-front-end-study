# -*- coding:utf-8 -*-
"""
# datetime: 17:21
# software: PyCharm
"""

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = '../datas/iris.data'
names = ['A', 'B', 'C', 'D', 'cla']
df = pd.read_csv(path, header=None, names=names, sep=",")


# print(df.head())
# print(df.shape)
# print(df["cla"].value_counts())
# sys.exit()


# 2. 数据清洗
# NOTE: 不需要做数据处理
def parse_record(row):
    result = []
    r = zip(names, row)
    for name, value in r:
        if name == 'cla':
            if value == 'Iris-setosa':
                result.append(1)
            elif value == 'Iris-versicolor':
                result.append(2)
            elif value == 'Iris-virginica':
                result.append(3)
            else:
                result.append(0)
        else:
            result.append(value)
    return result


df = df.apply(lambda row: pd.Series(parse_record(row), index=names), axis=1)
df.cla = df.cla.astype(np.int32)
df.info()
# print(df.cla.value_counts())
flag = False
# df = df[df.cla != 3]
# print(df.cla.value_counts())

# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df[names[0:-1]]
# print(X.shape)
Y = df[names[-1]]
# print(Y.value_counts())
# sys.exit()

# 4. 数据分割
# train_size: 给定划分之后的训练数据的占比是多少，默认0.75
# random_state：给定在数据划分过程中，使用到的随机数种子，默认为None，使用当前的时间戳；给定非None的值，可以保证多次运行的结果是一致的。
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.6, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 特征工程的操作
# NOTE: 不做特征工程

# 6. 模型对象的构建
"""
KNN:
    n_neighbors=5,
    weights='uniform',
    algorithm='auto', 
    leaf_size=30,
    p=2,
    metric='minkowski', 
    metric_params=None, 
    n_jobs=1
"""
KNN = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree')

# 7. 模型的训练
KNN.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = KNN.predict(x_train)
test_predict = KNN.predict(x_test)
print("KNN算法：测试集上的效果(准确率):{}".format(KNN.score(x_test, y_test)))
print("KNN算法：训练集上的效果(准确率):{}".format(KNN.score(x_train, y_train)))

# 模型的保存与加载
import joblib

joblib.dump(KNN, "./knn.m")  # 保存模型
# joblib.load() # 加载模型
