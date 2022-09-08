import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib as mpt
import sys

# 加载数据
data = pd.read_csv('./datas/boston_housing.data', sep='\s+', header=None)

# 获取特征属性X和目标属性Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

# 特征工程
'''
PolynomialFeatures ####多项式扩展
degree=2,扩展的阶数
interaction_only=False,是否只保留交互项
include_bias=True，是否需要偏置项
'''

print(x_train.shape)
print(x_test.shape)
print(x_test.iloc[0,:])
poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
print(x_train_poly.shape)
print(x_test_poly.shape)
# print(x_test_poly[0])
# sys.exit()
# 构建模型
# linear = LinearRegression(fit_intercept=True)
# lasso = Lasso(alpha=1000, fit_intercept=True)
ridge = Ridge(alpha=1.0, fit_intercept=True)
# 模型训练
# linear.fit(x_train_poly, y_train)
# lasso.fit(x_train_poly, y_train)
ridge.fit(x_train_poly, y_train)
# print(linear.coef_)
# print(linear.intercept_)
# print(lasso.coef_)
# print(lasso.intercept_)
print(ridge.coef_)
print(ridge.intercept_)
# 预测测试机
# y_test_hat = linear.predict(x_test_poly)
# y_test_hat = lasso.predict(x_test_poly)
y_test_hat = ridge.predict(x_test_poly)
print("-" * 100)
# print(linear.score(x_train_poly, y_train))
# print(linear.score(x_test_poly, y_test))
# print(lasso.score(x_train_poly, y_train))
# print(lasso.score(x_test_poly, y_test))
print(ridge.score(x_train_poly, y_train))
print(ridge.score(x_test_poly, y_test))
plt.plot(range(len(x_test)), y_test, 'r', label=u'true')
plt.plot(range(len(x_test)), y_test_hat, 'g', label=u'predict')
plt.legend(loc='upper right')
plt.show()

# todo：根据参数解析式写一个线性回归的伪代码，然后用波士顿房屋数据跟我们sklearn的库作比较
