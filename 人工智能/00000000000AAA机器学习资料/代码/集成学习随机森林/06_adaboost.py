import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

df = pd.DataFrame([[0, 1],
                   [1, 1],
                   [2, 1],
                   [3, -1],
                   [4, -1],
                   [5, -1],
                   [6, 1],
                   [7, 1],
                   [8, 1],
                   [9, -1]])

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

###第一个弱学习器
## 初始化样本权重
w1 = np.ones(df.shape[0]) / df.shape[0]
# print(w1)
# sys.exit()
##构造弱分类器G1
model1 = DecisionTreeClassifier(max_depth=1)
model1.fit(X, Y, sample_weight=w1)
###误差率
# print(w1[model1.predict(X)!=Y])
e1 = sum(w1[model1.predict(X) != Y])
# print(e1)
###弱学习器G1的权重α1
a1 = 0.5 * np.log((1 - e1) / e1)
# print(a1)
# sys.exit()
###第二个弱学习器G2
###更新样本权重值
w2 = w1 * np.exp(-a1 * Y * model1.predict(X))
# print(w2)
w2 = np.array(w2 / sum(w2))  ##归一化
# print(w2)
# sys.exit()
##训练模型G2
model2 = DecisionTreeClassifier(max_depth=1)
model2.fit(X, Y, sample_weight=w2)

###误差率e2
e2 = sum(w2[model2.predict(X) != Y])
print(e2)
###求G2的权重α2
a2 = 0.5 * np.log((1 - e2) / e2)
print(a2)
# f = a1*G1+a2*G2
# sys.exit()

###第三个弱学习器 G3
###更新样本权重值
w3 = w2 * np.exp(-a2 * Y * model2.predict(X))
# print(w3)
w3 = np.array(w3 / sum(w3))  ##归一化
print(w3)
###训练模型G3
model3 = DecisionTreeClassifier(max_depth=1)
model3.fit(X, Y, sample_weight=w3)

###误差率e3
e3 = sum(w3[model3.predict(X) != Y])
# print(e3)
###求G3的权重α3
a3 = 0.5 * np.log((1 - e3) / e3)
# print(a3)
# f = a1*G1+a2*G2+a3*G3


##最终分类器的线性组合f3
# f3 =a1*model1+a2*model2+a3*model3
## 最终的分类器G
# G = sign(f3)
##预测
y_hat = np.sign(a1 * model1.predict(X) + a2 * model2.predict(X) + a3 * model3.predict(X))
print(Y.tolist())
print(y_hat)

# todo:把这个代码整理成一个方法或者一个类，循环的方式；最大迭代次数、停止条件的误差率、决策树的超参、底数【2，e】
