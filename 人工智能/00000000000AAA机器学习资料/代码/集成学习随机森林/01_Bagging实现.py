import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

'''
bagging 回归
'''
###数据
df = pd.DataFrame([[1, 10.56],
                   [2, 27],
                   [3, 39.1],
                   [4, 40.4],
                   [5, 58],
                   [6, 60.5],
                   [7, 79],
                   [8, 87],
                   [9, 90],
                   [10, 95]],
                  columns=['X', 'Y'])
# print(df)
M = []  ###用来存储弱学习器
n_trees = 100  ###构造的弱学习器的数量

for i in range(n_trees):  ##循环训练我们的弱学习器
    ###对样本进行有放回的抽样m次
    '''
    sample() 抽样
    n=None,  抽样数据的条数
    frac=None, 抽样的比例
    replace=False, 是否有放回抽样
    weights=None, 权重
    random_state=None, 随机数种子
    axis=None 维度
    '''
    tmp = df.sample(frac=1.0, replace=True)  ###不需要设置随机数种子
    # tmp = tmp.drop_duplicates()  # ##去重
    X = tmp.iloc[:, :-1]
    Y = tmp.iloc[:, -1]
    model = DecisionTreeRegressor(max_depth=1)
    model.fit(X, Y)

    M.append(model)

###做预测
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

mode01 = DecisionTreeRegressor(max_depth=1)
mode01.fit(x, y)
y_hat_01 = mode01.predict(x)
print(y_hat_01)
print(mode01.score(x, y))
print("-" * 100)
res = np.zeros(df.shape[0])
for j in M:
    res += j.predict(x)
y_hat = res / n_trees
print(y_hat)
print('R2:', r2_score(y, y_hat))
