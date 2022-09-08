import numpy as np

# ## KNN加权投票--分类
# #初始化数据
T = [
    [3, 104, -1],
    [2, 100, -1],
    [1, 81, -1],
    [101, 10, 1],
    [99, 5, 1],
    [98, 2, 1]]
# #初始化待测样本
x = [18, 90]
# x = [3,104]
# x = [50, 50]
# #初始化邻居数
K = 3
# #初始化存储距离列表[[距离1，标签1],[距离2，标签2]....]
listDistance = []
# #循环每一个数据点，把计算结果放入dis
for i in T:
    dis = np.sum((np.array(i[:-1]) - np.array(x)) ** 2) ** 0.5  ##欧氏距离
    listDistance.append([dis, i[-1]])
# #对dis按照距离排序
listDistance.sort()

print(listDistance)
weight = [1/i[0] for i in listDistance[:K]]
print(weight)
weight /= sum(weight)
print(weight)

# pre = -1 if sum([1 / i[0] * i[1] for i in listDistance[:K]]) < 0 else 1
# print(pre)
