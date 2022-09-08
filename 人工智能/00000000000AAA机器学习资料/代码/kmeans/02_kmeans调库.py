import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 2], [2, 2], [6, 8], [7, 8]])
# C = np.array([[1, 2], [2, 2]])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
print(kmeans.cluster_centers_)  ##簇中心
print(kmeans.labels_)  ###簇的标签
print(kmeans.score(X))  ##
