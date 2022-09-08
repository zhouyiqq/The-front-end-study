import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpt

# 加载数据
data = pd.read_csv('./datas/boston_housing.data', sep='\s+', header=None)

# 数据预处理

# 获取特征属性X和目标属性Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# 划分训练集和测试集
xTarin, xTest, yTarin, yTest = train_test_split(X, Y, test_size=0.3, random_state=10)

# 构建模型
# fit_intercept是否需要截距项
linear = LinearRegression(fit_intercept=True)

# 模型训练
linear.fit(xTarin, yTarin)
print(linear.coef_)
print(linear.intercept_)
yPredict = linear.predict(xTest)
print(linear.score(xTarin, yTarin))
print(linear.score(xTest, yTest))
y_train_hat = linear.predict(xTarin)
# 训练集
plt.plot(range(len(xTarin)), yTarin, 'r', label=u'true')
plt.plot(range(len(xTarin)), y_train_hat, 'g', label=u'predict')
plt.legend(loc='upper right')
plt.show()

# 测试集
# plt.plot(range(len(xTest)), yTest, 'r', label=u'true')
# plt.plot(range(len(xTest)), yPredict, 'g', label=u'predict')
# plt.legend(loc='upper right')
# plt.show()
# todo：根据参数解析式写一个线性回归的伪代码，然后用波士顿房屋数据跟我们sklearn的库作比较