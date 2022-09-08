# -- encoding:utf-8 --
"""
Create by ibf on 14:33
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from sklearn.model_selection import train_test_split


from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

mpl.rcParams['font.sans-serif'] = [u'simHei']

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = 'household_power_consumption_1000_2.txt'
df = pd.read_csv(path, sep=';')

# 2. 数据清洗
df.replace('?', np.nan, inplace=True)
df = df.dropna(axis=0, how='any')

# 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
def date_format(dt):
    date_str = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return [date_str.tm_year, date_str.tm_mon, date_str.tm_mday, date_str.tm_hour, date_str.tm_min, date_str.tm_sec]

X = df.iloc[:, 0:2]
X = X.apply(lambda row: pd.Series(date_format(row)), axis=1)
Y = df.iloc[:, 4].astype(np.float32)

# 4. 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

##初始化基模型
linear  = LinearRegression(fit_intercept=True)
ridge = Ridge(alpha=0.1)
knn = KNeighborsRegressor(weights='distance')
rf = RandomForestRegressor(n_estimators=100,max_depth=3)

##组合模型/元模型
svr_rbf = SVR(kernel='rbf',gamma=0.1,C=0.1)
###stacking
stackingreg = StackingRegressor(regressors=[linear,ridge,knn,rf],meta_regressor=svr_rbf)



# params = {'linearregression__fit_intercept': [True,False],
#           'ridge__alpha': [0.01,0.1, 1.0, 10.0],
#           'kneighborsregressor__n_neighbors':[1,3,5,7,9],
#           'randomforestregressor__n_estimators':[50,100,150],
#           'randomforestregressor__max_depth':[1,3,5,7,9],
#           'meta_regressor__C': [0.1, 1.0, 10.0],
#           'meta_regressor__gamma': [0.1, 1.0, 10.0]}
#

# grid = GridSearchCV(estimator=stackingreg,
#                     param_grid=params,
#                     cv=5,
#                     refit=True)
# print(stackingreg.get_params().keys())
# """
#
# """
# # import sys
# # sys.exit(0)
# grid.fit(x_train, y_train)
# print("最优参数:{}".format(grid.best_params_))
""""

"""
# print("最优参数对应的最优模型:{}".format(grid.best_estimator_))
# print("最优模型对应的这个评估值:{}".format(grid.best_score_))

# ## 网格调参后进行训练
stackingreg.fit(x_train,y_train)
print(stackingreg.score(x_train,y_train))
print(stackingreg.score(x_test,y_test))

# ###画图展示
y_hat = stackingreg.predict(x_test)
plt.plot(range(len(x_test)),y_test,'r',label=u'真实值')
plt.plot(range(len(x_test)),y_hat,'b',label=u'测试值')
plt.legend()
plt.title('时间和电压回归预测')
plt.show()

