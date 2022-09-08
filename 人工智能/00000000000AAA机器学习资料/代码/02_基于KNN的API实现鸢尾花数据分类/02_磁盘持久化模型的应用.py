# -*- coding:utf-8 -*-
"""
# datetime: 20:29
# software: PyCharm
"""
import joblib

###1、加载回复模型
knn = joblib.load("./knn.m")

###2、对待预测的数据进行预测 （数据处理好后的数据）

x = [[5.1, 3.5, 1.4, 0.2]]
y_hat = knn.predict(x)
print(y_hat)
