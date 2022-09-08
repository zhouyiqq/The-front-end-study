import pandas as pd
import numpy as np
import sys
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler  ##标准化，归一化
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

pd.set_option("display.max_columns", None)

##读取数据
datas = pd.read_csv('./datas/risk_factors_cervical_cancer.csv', sep=',')
# print(datas.head())
# print(datas.info())
names = datas.columns
# print(names)
# sys.exit()
###数据清洗
datas.replace('?', np.nan, inplace=True)
# print(datas.info())
# sys.exit()
# print(datas.head())
###使用Imputer进行缺省值的填充 列填充
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = SimpleImputer()
datas = imputer.fit_transform(datas)
datas = pd.DataFrame(datas, columns=names)
# print(datas.head())
# print(datas.info())

###获取特征属性X 和目标属性Y
X = datas.iloc[:, :-4]
Y = datas.iloc[:, -4:].astype('int')
# print(X.info())
# print(Y.info())

###数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

###构建一个管道
###数据标准化，数据归一化 （数据量纲） 决策树来说我们其实不需要做这个操作
##标准化：把数据转化为均值为0，方差为1的
##归一化：把数据压缩到0-1

###PCA降维
models = [Pipeline([('standarscaler', StandardScaler()),
                    ('pca', PCA()),
                    ('RF', RandomForestClassifier())]),
          Pipeline([
              ('pca', PCA(n_components=0.5)),
              ('RF', RandomForestClassifier(n_estimators=50, max_depth=1))])
          ]
'''
###设置参数
params = {'pca__n_components':[0.5,0.6,0.7,0.8,0.9],
          'RF__n_estimators':[50,100,150],
          'RF__max_depth':[1,3,5,7]}
##网格调参
model = GridSearchCV(estimator=models[0],param_grid=params,cv=5)
##训练
model.fit(x_train,y_train)
print('最优参数：',model.best_params_)
print('最优模型：',model.best_estimator_)
print('最优模型的分数：',model.best_score_)
'''

model = models[1]
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

# ###保存模型
# from sklearn.externals import joblib
# joblib.dump(model,'./model/risk01.m')
