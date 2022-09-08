import numpy as np


class Linear:
    def __init__(self, b=True):
        self.b = b
        self.theta = None
        self.theta0 = 0

    def train(self, X, Y):
        if self.b:
            X = np.column_stack((np.ones_like(X), X))
        # 二、为了求解比较方便，将numpy的'numpy.ndarray'的数据类型转换为矩阵的形式的。
        X = np.mat(X)
        Y = np.mat(Y)
        # 三、根据解析式的公式求解theta的值
        theta = (X.T * X).I * X.T * Y
        if self.b:
            self.theta0 = theta[0]
            self.theta = theta[1:]
        else:
            self.theta0 = 0
            self.theta = theta

    def predict(self, X):
        predict_y = X * self.theta + self.theta0
        return predict_y

    def score(self, X, Y):
        # mse
        # r^2
        # mae

        pass

    def save(self):
        """

        :return:
        """
        # self.theta0
        # self.theta


if __name__ == '__main__':
    X1 = np.array([10, 15, 20, 30, 50, 60, 60, 70]).reshape((-1, 1))
    Y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))
    linear = Linear(b=True)
    linear.train(X1, Y)
    x_test = [[55]]
    y_test_hat = linear.predict(x_test)
    print(y_test_hat)
    print(linear.theta)
    print(linear.theta0)
