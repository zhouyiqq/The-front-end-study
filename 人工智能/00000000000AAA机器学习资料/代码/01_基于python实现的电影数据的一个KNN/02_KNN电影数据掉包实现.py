# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import neighbors

T = [
    [3, 104, -1],
    [2, 100, -1],
    [1, 81, -1],
    [101, 10, 1],
    [99, 5, 1],
    [98, 2, 1]]
##初始化待测样本
x = [[18, 90]]
##初始化邻居数
K = 3

data = pd.DataFrame(T, columns=['A', 'B', 'label'])
# print(data)
X_train = data.iloc[:, :-1]
# print(X_train)
Y_train = data.iloc[:, -1]
# print(Y_train)

KNN01 = neighbors.KNeighborsClassifier(n_neighbors=3)
KNN01.fit(X_train, Y_train)
y_predict = KNN01.predict(x)
print(y_predict)

score = KNN01.score(X=X_train, y=Y_train)
print(score)

KNN02 = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
KNN02.fit(X_train, Y_train)
y_predict = KNN02.predict(x)
print(y_predict)

##初始化数据
T = [
    [3, 104, 98],
    [2, 100, 93],
    [1, 81, 95],
    [101, 10, 16],
    [99, 5, 8],
    [98, 2, 7]]
##初始化待测样本
x = [[18, 90]]
# x =[[50, 50]]
##初始化邻居数
K = 5

data = pd.DataFrame(T, columns=['A', 'B', 'label'])
# print(data)
X_train = data.iloc[:, :-1]
# print(X_train)
Y_train = data.iloc[:, -1]
# print(Y_train)


# In[23]:


KNN03 = neighbors.KNeighborsRegressor(n_neighbors=K)
KNN03.fit(X_train, Y_train)
y_predict = KNN03.predict(x)
print(y_predict)

# In[24]:


KNN04 = neighbors.KNeighborsRegressor(n_neighbors=K, weights='distance')
KNN04.fit(X_train, Y_train)
y_predict = KNN04.predict(x)
print(y_predict)
