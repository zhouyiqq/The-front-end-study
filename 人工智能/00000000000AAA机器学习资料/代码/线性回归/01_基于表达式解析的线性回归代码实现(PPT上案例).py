# -- encoding:utf-8 --
"""
Create on 19/3/2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 一、构造数据
X1 = np.array([10, 15, 20, 30, 50, 60, 60, 70]).reshape((-1, 1))
Y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))

# 添加一个截距项对应的X值 np.column_stack()
# X = np.hstack((np.ones_like(X1), X1))
X = np.column_stack((np.ones_like(X1), X1))
# 不加入截距项
# X = X1
# print(X)
# print(Y)
# sys.exit()
# 二、为了求解比较方便，将numpy的'numpy.ndarray'的数据类型转换为矩阵的形式的。
X = np.mat(X)
Y = np.mat(Y)

# 三、根据解析式的公式求解theta的值
theta = (X.T * X).I * X.T * Y
print(theta)
# sys.exit()
# 四、 根据求解出来的theta求出预测值
predict_y = X * theta
x_test = [[1, 55]]
y_test_hat = x_test * theta
print("价格：", y_test_hat)

# print(predict_y)
# 四、画图可视化
plt.plot(X1, Y, 'bo', label=u'真实值')
plt.plot(X1, predict_y, 'r--o', label=u'预测值')
plt.legend(loc='lower right')
plt.show()

# 基于训练好的模型参数对一个未知的样本做一个预测
x = np.mat(np.array([[1.0, 55.0]]))
pred_y = x * theta
print("当面积为55平的时候，预测价格为:{}".format(pred_y))

##TODO: 两个特征的时候的预测 【房屋面积55，房间数量2】时的预测结果 注意画图画3D图
