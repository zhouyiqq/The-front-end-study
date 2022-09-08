import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

###数据
df = pd.DataFrame(
    [[1, 5.56], [2, 5.7], [3, 5.91], [4, 6.4], [5, 6.8], [6, 7.05], [7, 8.9], [8, 8.7], [9, 9], [10, 9.05]],
    columns=['X', 'Y'])
# df = pd.read_csv('./datas/boston_housing.data', sep='\s+', header=None)
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
y = Y  ##保留原始的Y

###Fm = F0+v*f1+v*f2+v*f3+...v*fm

###F0
F0 = np.mean(Y)
M = [F0]
##f1第一棵树 标签（label） Y-F0
Y = Y - F0 ##残差作为梯度，回归（平方和损失）
n_trees = 100
learning_rate = 0.1
for i in range(n_trees):
    model = DecisionTreeRegressor(max_depth=1)
    model.fit(X, Y)
    Y = Y - learning_rate * model.predict(X)
    M.append(model)

# print(M)

###预测
res = np.zeros(df.shape[0])
for j in range(len(M)):
    if j == 0:
        res += M[j]
        # print(res)
    else:
        res += learning_rate * M[j].predict(X)
        # print(res)
print(res)
y_hat = res
print(r2_score(y, y_hat))
