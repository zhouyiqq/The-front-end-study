import numpy as np
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report

# datas = fetch_20newsgroups_vectorized(data_home='./datas',
#                                       remove=('headers','footers','quotes'),
#                                       subset='train',
#                                       return_X_y=False)
datas = fetch_20newsgroups(data_home='./datas',
                           subset='train',
                           remove=('headers', 'footers', 'quotes'),
                           return_X_y=False
                           )
# # print(datas)
# print(len(datas.data))
# print(datas.target.shape)
# print(len(datas.target_names))
# print(datas.data[0])
# print(datas.target[0])
# print(datas.target_names[0])
# # print(1/0)
tfidf = TfidfVectorizer()
datas_tfidf = tfidf.fit_transform(datas.data)
# print(datas_tfidf.shape)
# print(datas_tfidf[0])

X = datas_tfidf
print(X.shape)
Y = datas.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# select = SelectKBest(score_func=chi2, k=2000)
# x_train = select.fit_transform(X=x_train, y=y_train)
# x_train = x_train.toarray()
# x_test = select.transform(x_test)
# x_test = x_test.toarray()

# pca = PCA(n_components=200)
# x_train = pca.fit_transform(x_train.toarray())
# x_test = pca.transform(x_test)

# # 词向量降维
svd = TruncatedSVD(n_components=1000, random_state=28)
x_train = svd.fit_transform(x_train)
x_test = svd.transform(x_test)

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_train_hat = gaussian.predict(x_train)
y_test_hat = gaussian.predict(x_test)
print(classification_report(y_train, y_train_hat))
print("------------------------")
print(classification_report(y_test, y_test_hat))
