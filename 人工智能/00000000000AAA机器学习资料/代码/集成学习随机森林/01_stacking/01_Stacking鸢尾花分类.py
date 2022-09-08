import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'simHei']

import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlxtend.classifier import StackingClassifier

from mlxtend.feature_selection import ColumnSelector

###1、读取数据
datas = pd.read_csv('iris.data', sep=',', header=None, names=['X1', 'X2', 'X3', 'X4', 'Y'])
# print(datas.head())
# print(datas.info())

### 2、数据清洗

### 3、获取特征属性X和目标属性Y
X = datas.iloc[:, :-1]
Y = datas.iloc[:, -1]
# print(X.shape)
# print(Y.shape)
# print(Y.value_counts())  ##看下目标属性的值

# LabelEncoder  0,1, 2
labelencoder = LabelEncoder()
# print(Y.ravel())
Y = labelencoder.fit_transform(Y)
# print(Y)

### 4、分割数据集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=28)
# print(x_train.shape)
# print(y_train.shape)

### 5、特征工程

### 模型构建
# a、构造基学习器 knn、RF、softmax、GBDT。。。。
knn = KNeighborsClassifier(n_neighbors=7)
softmax = LogisticRegression(C=0.1, solver='lbfgs', multi_class='multinomial', fit_intercept=False)
gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)
rf = RandomForestClassifier(max_depth=5, n_estimators=150)

# b、元学习器
lr = LogisticRegression(C=0.1, solver='lbfgs', multi_class='multinomial')

### stacking学习器
'''
1、最基本的使用方法，用前面基学习器的输出作为元学习器的输入
2、使用基学习器的输出类别的概率值作为元学习器输入，use_probas=True,若average_probas=True，那么这些基分类器对每一个类别产生的概率进行平均，否者直接拼接
 classifier1  = [0.2,0.5,0.3]
 classifier2  = [0.3,0.3,0.4]
  average_probas=True: [0.25,0.4,0.35]
  average_probas=Flase: [0.2,0.5,0.3,0.3,0.3,0.4]
  
3、对训练集中的特征维度进行操作，每次训练不同的基学习器的时候用不同的特征，比如我再训练KNN的时候只用前两个特征，训练RF的时候用其他的几个特征
    通过pipline来实现
'''
'''
classifiers, 基学习器
meta_classifier, 元学习器
use_probas=False, 
drop_last_proba=False,
average_probas=False, 
verbose=0,
use_features_in_secondary=False,
store_train_meta_features=False,
use_clones=True
'''

###方式一
stacking01 = StackingClassifier(classifiers=[knn, softmax, gbdt, rf],
                                meta_classifier=lr)

###方式二
stacking02 = StackingClassifier(classifiers=[knn, softmax, gbdt, rf],
                                meta_classifier=lr,
                                use_probas=True,
                                average_probas=False)

###方式三
# 基学习器
pipe_knn = Pipeline([('x', ColumnSelector([0, 1])),
                     ('knn', knn)])
pipe_softmax = Pipeline([('x', ColumnSelector([2, 3])),
                         ('softmax', softmax)])
pipe_rf = Pipeline([('x', ColumnSelector([0, 3])),
                    ('rf', rf)])
pipe_gbdt = Pipeline([('x', ColumnSelector([1, 2])),
                      ('gbdt', gbdt)])
##stacking
stacking03 = StackingClassifier(classifiers=[pipe_knn, pipe_softmax, pipe_rf, pipe_gbdt],
                                meta_classifier=lr)

###模型训练与比较
scores_train = []
scores_test = []
models = []
times = []

for clf, modelname in zip([knn, softmax, gbdt, rf, stacking01, stacking02, stacking03],
                          ['knn', 'softmax', 'gbdt', 'rf', 'stacking01', 'stacking02', 'stacking03']):
    print('start:%s' % (modelname))
    start = time.time()
    clf.fit(x_train, y_train)
    end = time.time()
    print('耗时：{}'.format(end - start))
    score_train = clf.score(x_train, y_train)
    score_test = clf.score(x_test, y_test)
    scores_train.append(score_train)
    scores_test.append(score_test)
    models.append(modelname)
    times.append(end - start)
print('scores_train:', scores_train)
print('scores_test', scores_test)
print('models:', models)
print('开始画图----------')
plt.figure(num=1)
plt.plot([0, 1, 2, 3, 4, 5, 6], scores_train, 'r', label=u'训练集')
plt.plot([0, 1, 2, 3, 4, 5, 6], scores_test, 'b', label=u'测试集')
plt.title(u'鸢尾花数据不同分类器准确率比较', fontsize=16)
plt.xticks([0, 1, 2, 3, 4, 5, 6], models, rotation=0)
plt.legend(loc='lower left')
plt.figure(num=2)
plt.plot([0, 1, 2, 3, 4, 5, 6], times)
plt.title(u'鸢尾花数据不同分类器训练时间比较', fontsize=16)
plt.xticks([0, 1, 2, 3, 4, 5, 6], models, rotation=0)
plt.legend(loc='lower left')
plt.show()
