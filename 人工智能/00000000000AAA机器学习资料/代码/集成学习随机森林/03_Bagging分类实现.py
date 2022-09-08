# _*_coding:utf-8_*_
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

df = pd.DataFrame([[0, 1], [1, 1], [2, 1], [3, -1], [4, -1],
                   [5, -1], [6, 1], [7, 1], [8, 1], [9, -1]])
n_tree = 199
models = []

for i in range(n_tree):
    df2 = df.sample(frac=1.0, replace=True)
    X = df2.iloc[:, :-1]
    Y = df2.iloc[:, -1]
    dec = DecisionTreeClassifier(max_depth=1)
    dec.fit(X, Y)
    models.append(dec)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]
total = np.zeros(df.shape[0])
for i in range(n_tree):
    total += np.array(models[i].predict(x))
print(total)
y_hat = np.sign(total)
print(y_hat)
print(accuracy_score(y, y_hat))
print(f1_score(y, y_hat))
print("-" * 100)
model01 = DecisionTreeClassifier(max_depth=1)
model01.fit(x, y)
y_hat_01 = model01.predict(x)
print(y_hat_01)
print(accuracy_score(y, y_hat_01))
print(f1_score(y, y_hat_01))
