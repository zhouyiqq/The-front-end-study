# -*- coding:utf-8 -*-
"""
#@TIME: 13:08
#@FILE: 03_KNN伪代码.PY
#@Author: lq
"""

'''
简单实现一下等权分类 封装成KNN类
'''
import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score


class KNN():
    '''
    KNN的步骤：
    1、从训练集合中获取K个离待预测样本距离最近的样本数据；
    2、根据获取得到的K个样本数据来预测当前待预测样本的目标属性值
    '''

    def __init__(self, k, with_kd_tree=True):
        self.k = k
        self.with_kd_tree = with_kd_tree

    def fit(self, x, y):
        '''
        fit 训练模型 保存训练数据
        如果with_kd_tree=True 则训练构建kd_tree
        :param x:
        :param y:
        :return:
        '''
        ###将数据转化为numpy数组的形式
        x = np.asarray(x)
        y = np.asarray(y)
        self.train_x = x
        self.train_y = y
        if self.with_kd_tree:
            self.kd_tree = KDTree(x, leaf_size=10, metric='minkowski')

    def fetch_k_neighbors(self, x):
        '''
        ## 1、从训练集合中获取K个离待预测样本距离最近的样本数据；
        ## 2、根据获取得到的K个样本数据来预测当前待预测样本的目标属性值
        :param x: 当前样本的特征属性x(一条样本)
        :return:
        '''
        if self.with_kd_tree:
            ## kd_tree.query([x],k=self.k,return_distance=True))
            # 返回对应最近的k个样本的下标，如果return_distance=True同时也返回距离
            # print(self.kd_tree.query([x],k=self.k,return_distance=True)[0])

            # 获取对应最近k个样本的标签
            index = self.kd_tree.query([x], k=self.k, return_distance=False)[0]
            # print(index)
            k_neighbors_label = []
            for i in index:
                k_neighbors_label.append(self.train_y[i])
            # print(k_neighbors_label)
            return k_neighbors_label
        else:
            ## 定义一个列表用来存储每个样本的距离以及对应的标签
            # [[距离1,标签1],[距离2,标签2],[距离3,标签3]....]
            listDistance = []
            for index, i in enumerate(self.train_x):
                dis = np.sum((np.array(i) - np.array(x)) ** 2) ** 0.5
                listDistance.append([dis, self.train_y[index]])
            # print(listDistance)

            ## 按照dis对listDistance进行排序
            # listDistance.sort()
            # print(listDistance)
            sort_listDistance = np.sort(listDistance, axis=0)
            # print(sort_listDistance)
            # print(type(sort_listDistance))

            ## 获取取前K个最近距离的样本的标签
            k_neighbors_label = sort_listDistance[:self.k, -1]
            # print(k_neighbors_label)
            ## 也可以获取前k个最近邻居的距离
            k_neighbors_dis = sort_listDistance[:self.k, :-1]
            return k_neighbors_label

    def predict(self, X):
        '''
        模型预测
        :param X: 待预测样本的特征矩阵（多个样本）
        :return: 预测结果
        '''
        ### 将数据转化为numpy数组的格式
        X = np.asarray(X)

        ## 定义一个列表接收每个样本的预测结果
        result = []
        for x in X:
            k_neighbors_label = self.fetch_k_neighbors(x)
            ### 统计每个类别出现的次数
            y_count = pd.Series(k_neighbors_label).value_counts()
            # print(y_count)
            ### 产生结果
            y_ = y_count.idxmax()
            # y_ = y_count.argmax() ##idxmax() 和 argmax 功能一样，获取最大值对应的下标索引
            result.append(int(y_))
        return result

    def socre(self, x, y):
        '''
        模型预测得分：我们使用准确率 accuracy_score
        :param x:
        :param y:
        :return:
        '''
        y_true = y
        y_pred = self.predict(x)
        return accuracy_score(y_true, y_pred)


if __name__ == '__main__':
    T = np.array([
        [3, 104, -1],
        [2, 100, -1],
        [1, 81, -1],
        [101, 10, 1],
        [99, 5, 1],
        [98, 2, 1]])
    X_train = T[:, :-1]
    Y_train = T[:, -1]
    x_test = [[18, 90], [50, 10]]
    knn = KNN(k=5, with_kd_tree=True)
    knn.fit(x=X_train, y=Y_train)
    print(knn.socre(x=X_train, y=Y_train))
    # knn.fetch_k_neighbors(x_test[0])
    print('预测结果：{}'.format(knn.predict(x_test)))
    print('-----------下面测试一下鸢尾花数据-----------')
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, Y = load_iris(return_X_y=True)
    print(X.shape, Y.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print(x_train.shape, y_train.shape)
    knn01 = KNN(k=3, with_kd_tree=True)
    knn01.fit(x_train, y_train)
    print(knn01.socre(x_train, y_train))
    print(knn01.socre(x_test, y_test))
