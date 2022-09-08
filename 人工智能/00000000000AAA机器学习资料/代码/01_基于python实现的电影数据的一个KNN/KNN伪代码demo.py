'''
简单实现一下等权分类 封装成KNN类
    实现fit，predict，score方法

'''
import numpy as np
import pandas as pd


class KNN:
    '''
    KNN的步骤：
    1、从训练集合中获取K个离待预测样本距离最近的样本数据；
    2、根据获取得到的K个样本数据来预测当前待预测样本的目标属性值
    '''

    def __init__(self, k):
        self.k = k
        pass

    def fit(self, x, y):
        '''
        训练模型  实际上就是存储数据
        :param x: 训练数据x
        :param y: 训练数据y
        :return:
        '''
        ### 将数据转化为numpy数组的形式进行存储
        self.train_x = np.array(x)
        self.train_y = np.array(y)

    def feach_k_neighbors(self, x):
        '''
        # 1、从训练集合中获取K个离待预测样本距离最近的样本数据；
        # 2、根据获取得到的K个样本数据来预测当前待预测样本的目标属性值
        :param x:待预测的一条数据
        :return: 最近k个邻居的label
        '''
        ###列表 [[dis1，标签1]，[dis2，标签2].。。。。。。]
        listdistance = []
        ##循环每一个数据，计算他的dis
        for index, i in enumerate(self.train_x):  ## t是每条电影的数据
            # print(index)
            dis = np.sum((np.array(i) - np.array(x)) ** 2) ** 0.5
            listdistance.append([dis, self.train_y[index]])
        # print(listdistance)
        ##按照dis进行排序
        listdistance.sort()
        # print(listdistance)
        # sys.exit()

        ##选取K个邻居放入投票箱
        # print(listdistance[:self.k])
        arr = np.array(listdistance[:self.k])[:, -1]
        # print(arr)
        return arr

    def predict(self, x):
        '''
        对待预测数据进行预测
        :param x: 待预测数据的特征属性x 是个矩阵
        :return: 所有数据的预测label
        '''
        ### 将数据转化为numpy数组的形式
        self.pre_x = np.array(x)

        # 遍历每一条带预测数据
        Y_pre = []
        for x in self.pre_x:
            # print(x)
            # 1、从训练集合中获取K个离待预测样本距离最近的样本数据；
            k_nearst_neighbors_label = self.feach_k_neighbors(x)

            # 2、根据获取得到的K个样本数据来预测当前待预测样本的目标属性值
            ##统计投票
            a = pd.Series(k_nearst_neighbors_label).value_counts()
            # print(a)
            # pre = a.idxmax()  ##idxmax() 和 argmax 功能一样，获取最大值对应的下标索引
            y_pre = a.idxmax()
            # pre = a.argmax()
            # print(pre)
            Y_pre.append(y_pre)
        return Y_pre

    def score(self, x, y):
        '''

        :param x:
        :param y:
        :return: 准确率
        '''
        y_hat = self.predict(x)
        acc = np.mean(y == y_hat)
        return acc


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
    x_test = [[18, 90], [50, 50]]
    knn = KNN(k=5)
    knn.fit(x=X_train, y=Y_train)
    print(knn.predict(X_train))
    print(knn.predict(x_test))
    print(knn.score(x=X_train, y=Y_train))
    # knn.fetch_k_neighbors(x_test[0])
    print('预测结果：{}'.format(knn.predict(x_test)))
    print('-----------下面测试一下鸢尾花数据-----------')
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, Y = load_iris(return_X_y=True)
    print(X.shape, Y.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print(x_train.shape, y_train.shape)
    knn01 = KNN(k=1)
    knn01.fit(x_train, y_train)
    print(knn01.score(x_train, y_train))
    print(knn01.score(x_test, y_test))
