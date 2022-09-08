import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
datas = pd.read_csv('../datas/crx.data', sep=',', header=None, names=names)
# print(datas.info())
# print(datas.head())

X = datas.iloc[:, :-1]
Y = datas.iloc[:, -1]
# column_names = X.columns  ###获取列名
###数据预处理
# 获取离散特征
X_lisan = X[['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']]
# print(X_lisan.info())
X_lisan = pd.get_dummies(X_lisan)
# print(X_lisan.info())
#
# ### 连续数据的处理
X_lianxu = X[['A2', 'A3', 'A8', 'A11', 'A14', 'A15']]
# print(X_lianxu.info())
X_lianxu.replace('?', np.nan, inplace=True)
# # print(X_lianxu)
# ###数据填充的一种方法
from sklearn.impute import SimpleImputer
#
# '''
# strategy:"mean"均值填充   "median" 中值填充  "most_frequent"  众数填充
# '''
impt = SimpleImputer(missing_values=np.nan, strategy="mean")
X_lianxu = impt.fit_transform(X_lianxu)
# # print(X_lianxu.shape)
# # print(type(X_lianxu))
X_lianxu = pd.DataFrame(X_lianxu, columns=['A2', 'A3', 'A8', 'A11', 'A14', 'A15'])
# # print(X_lianxu.info())
#
# ##合并连续特征和离散特征
X = pd.concat([X_lianxu, X_lisan], axis=1)
print(X.info())
column_names = X.columns  ###获取列名
print(column_names)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=110)

dectree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, min_samples_split=15)
dectree.fit(x_train, y_train)
print(dectree.score(x_train, y_train))
print(dectree.score(x_test, y_test))
print(dectree.feature_importances_)
import pydotplus
from sklearn import tree

dot_data = tree.export_graphviz(dectree,
                                out_file=None,
                                feature_names=column_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('crx03.png')
