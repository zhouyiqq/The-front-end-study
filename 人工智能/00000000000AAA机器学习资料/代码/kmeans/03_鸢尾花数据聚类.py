import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

datas = pd.read_csv('./datas/iris.data', sep=",", header=None)
print(datas.head())
X = datas.iloc[:, :-1]

clf = KMeans(n_clusters=4)
clf.fit(X)
print(clf.cluster_centers_)
print(clf.labels_)
