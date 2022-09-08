import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

###数据
df = pd.DataFrame([[1, 5.56],
                   [2, 5.7],
                   [3, 5.91],
                   [4, 6.4],
                   [5, 6.8],
                   [6, 7.05],
                   [7, 8.9],
                   [8, 8.7],
                   [9, 9],
                   [10, 9.05]], columns=['X', 'Y'])
# df = pd.read_csv('./datas/boston_housing.data', sep='\s+', header=None)
X1 = df.iloc[:, :-1]
Y1 = df.iloc[:, -1]
X, x_test, Y, y_test = train_test_split(X1, Y1, test_size=0.2, random_state=10)
y = Y  ###保留原始的Y

M = []
n_trees = 10
learningrate = 0.1
for i in range(n_trees):
    model = DecisionTreeRegressor(max_depth=1)
    model.fit(X, y)
    y = y - model.predict(X)
    # print(Y)
    # print(id(Y))
    # print(id(y))
    M.append(model)
    # print(model.predict(X))
    # print(Y.ravel())

###预测训练集
res = np.zeros(X.shape[0])
for j in M:
    res += j.predict(X)

y_hat = res
print(Y)
print(y_hat)
print(r2_score(Y, y_hat))
###预测测试集
res_test = np.zeros(x_test.shape[0])
for j in M:
    res_test += j.predict(x_test)

y_test_hat = res_test
print(y_test)
print(y_test_hat)
print(r2_score(y_test, y_test_hat))
