import numpy as np
import sys
X = np.array([[1.0, 2.0], [2.0, 2.0], [6.0, 8.0], [7.0, 8.0]])
# C = np.array([[1.0, 2.0], [2.0, 2.0]])
C = np.array([[2.0, 2.0], [1.0, 2.0]])
iters = 5 ###迭代次数
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
        # print(A)
        # sys.exit()
    A = np.array(A)
    print(A)
    ###将样本分配的所属的聚类中心（获取最小值的下标）
    minidx = np.argmin(A, axis=0)
    print(minidx)

    ###获取每个簇的样本
    for i in range(len(C)):
        a = X[minidx == i]  ###每个簇的样本
        print('a\n', a)
        C[i] = np.mean(a, axis=0)  ###更新我们的聚类中心
        # print(C[i])
    print("C\n", C)
    # todo: 就是C不再更新停止
    print('---------------------------------')
    iters -= 1

# todo：封装成类【伪代码】，加上就是C不再更新停止