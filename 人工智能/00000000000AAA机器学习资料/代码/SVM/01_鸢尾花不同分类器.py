import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  ##KNN
from sklearn.linear_model import LogisticRegression  ##逻辑回归
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl

import time

import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('iris.data', header=None)
# print(data.head())
X = data.iloc[:, :2]
Y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

svc = SVC(C=0.2, kernel="rbf",decision_function_shape="ovr")
knn = KNeighborsClassifier(n_neighbors=5)
log = LogisticRegression()
rand = RandomForestClassifier(n_estimators=150, max_depth=3)
models = np.array([svc, knn, log, rand])
T = []
train_scores = []
test_scores = []
for i in models:
    N = time.clock()
    i.fit(x_train, y_train)
    M = time.clock()
    T.append(M - N)
    train_scores.append(i.score(x_train, y_train))
    test_scores.append(i.score(x_test, y_test))

plt.figure(num=1)
plt.plot(['01svc', '02knn', '03log', '04rand'], test_scores, 'r-', label='test_score')
plt.plot(['01svc', '02knn', '03log', '04rand'], train_scores, 'g-o', label='train_score')
plt.ylim(0.5, 1.01)
plt.figure(num=2)
plt.plot(['01svc', '02knn', '03log', '04rand'], T, label='time')
plt.show()
