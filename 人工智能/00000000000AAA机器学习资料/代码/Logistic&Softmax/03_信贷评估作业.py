import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import sys
pd.set_option('display.max_columns', None)
# 对数据列进行分类 比如第一列有ab？三种，作为列表返回
def classify(data):
    data1Class = data.value_counts()
    data1Class = list(data1Class.index)
    return data1Class

# 根据classify返回的类型列表，进行亚编码转换返回
def oneHot(data1Class,data1):
    '''

    :param data1Class:
    :param data1:
    :return:
    '''
    data11 = []
    for index in range(len(data1)):
        data = []
        for cla in data1Class:
            if data1[index] == cla:
                data.append(1)
            else:
                data.append(0)
        data11.append(data)
    return data11

# print(oneHot(['a','b','c'],['a','a','c','b','c']))
# sys.exit(0)

# 将亚编码列表合并到X中返回
def merge(datas,data1):
    '''

    :param datas: 原数据
    :param data1: 要处理的onehot的那一列
    :return: datas 把onehot加入的新数据
    '''
    data2 = pd.DataFrame(oneHot(classify(data1), data1))
    lenght = len(data2.columns)
    data2.columns = range(50,50+lenght,1)
    datas = pd.concat([datas, data2], axis=1)
    return datas

datas = pd.read_csv('./datas/crx.data',header=None)
# print(datas.info())
###数据清洗   填充
datas.replace('?',np.nan,inplace=True)
datas = datas.dropna()
datas[1]=datas[1].astype(np.float)
# datas[13]=datas[13].astype(np.int)
print(datas.info())

# print(datas.head())
X = datas.iloc[:,:-1]
Y = datas.iloc[:,-1]


###数据填充，0，均值，众数  20经验
X[1] = X[1].replace('?',20.00)
###LabelEncoder
Y = Y.replace('+',0)
Y = Y.replace('-',1)

X = pd.get_dummies(X) ###做哑编码的 只对非数值型的数据做哑编码
print(X.head())

# #将没一列的数据检测，如果为Object，则进行亚编码转换
# for colunm in range(15):
#     if colunm == 1:
#         continue
#     if X[colunm].dtype == 'object':
#         data1 = X[colunm]
#         del X[colunm]
#         X = pd.DataFrame(merge(X, data1))

xTrain,xTest,yTrain,yTest = train_test_split(X,Y,test_size=0.3,random_state=10)

logist = LogisticRegression(penalty='l1')

logist.fit(xTrain,yTrain)
yPredict = logist.predict(xTest)

print(logist.score(xTest,yTest))

plt.scatter(range(len(xTest)),yTest,c = 'r',s = 8,zorder = 3,label = 'true')
plt.scatter(range(len(xTest)),yPredict,c = 'g',s = 10,zorder = 2,label = 'predict')
plt.show()
