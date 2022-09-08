import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv('iris.data', header=None)
# print(data.head())

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

label = LabelEncoder()
Y = label.fit_transform(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

gaussian = GaussianNB()
bernoull = BernoulliNB()
multin = MultinomialNB()

list_A = [gaussian, bernoull, multin]

one_test = []
train_score = []

for one in list_A:
    one.fit(x_train, y_train)
    one_test.append(one.score(x_test, y_test))
    train_score.append(one.score(x_train, y_train))
    # one.score(x_train, y_train)
    # y_hat = one.predict(x_train)
    ####各种错误
    # y_hat = one.predict(y_train)
    # one.score(x_train,y_hat)
    # one.score(y_train,y_hat)
    ###正确
    # accuracy_score(y_hat,y_train)
print(one_test)
print(train_score)
