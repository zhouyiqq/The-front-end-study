import numpy as np
import pandas as pd
import random


class Kmeans():
    cluster_centers_ = []
    labels_ = []

    def __init__(self, k, iters=100):
        self.k = k
        self.iters = iters

    def fit(self, X):
        ##随机获取K个初始化质心
        C = []
        for i in range(self.k):
            n = random.choices(list(range(X.shape[0])))
            c = np.array(X.iloc[n, :])[0]
            C.append(c)
        C = np.array(C)
        iters = self.iters  ###迭代次数
        while iters > 0:
            ###计算每个样本到每个聚类中心的距离
            A = []
            for c in C:  ##聚类中心
                # print(c)
                # print(X-c)
                # print(np.sum((X-c)**2,axis=1))
                a = np.sqrt(np.sum((X - c) ** 2, axis=1))
                # print(a)
                A.append(a)
            A = np.array(A)
            ###将样本分配的所属的聚类中心（获取最小值的下标）
            minidx = np.argmin(A, axis=0)
            ###获取每个簇的样本
            for i in range(len(C)):
                a = X[minidx == i]  ###每个簇的样本
                # print('a',a)
                C[i] = np.mean(a, axis=0)  ###更新我们的聚类中心
            iters -= 1
        # print(C)
        self.cluster_centers_ = C
        # print(self.cluster_centers_)
        self.labels_ = minidx

    def predict(self, x):
        ###获取训练后的聚类中心
        ###计算待测样本与聚类中心的距离
        ###归到距离最近的那个簇
        pass

    def sse(self, x):
        """
        评估指标
        :param x:
        :return:
        """
        pass


if __name__ == '__main__':
    # 获取数据
    datas = pd.read_csv('./datas/iris.data', header=None, names=['X1', 'X2', 'X3', 'X4', 'Y'])

    # 获取特征属性X
    X = datas.iloc[:, :-2]
    # print(X)
    kmeans = Kmeans(k=3)
    kmeans.fit(X)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)
