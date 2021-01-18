# -*-coding=utf-8-*-

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

class LogisticRegression(object):
    def __init__(self, lr=0.001, max_iter=100, n_class=2):
        self.learning_rate = lr
        self.max_iter = max_iter
        self.n_class = n_class
        self.w = []
        self.intercept = None

    def fit(self, X, y):
        """

        :param X: np.array (N, dim)
        :param y: np.array (n_class, )
        :return:
        """

        self.w = np.zeros(X.shape[1])
        self.intercept = np.random.rand()

        for epoch in range(self.max_iter):
            print("epoch "+ str(epoch+1))
            m = X.shape[0]
            y_pred = self.sigmoid(np.matmul(X, self.w) + self.intercept)
            # loss = loss(y[i], y_pred)
            dw = np.sum((X.T * np.array(y_pred - y)).T, axis=0) / m
            db = np.sum(y_pred-y) / m
            # print(dw)
            # print(self.w)
            self.w += -self.learning_rate * dw
            self.intercept += -self.learning_rate * db

            y_ = lr.predict(X)
            pred_label = [1 if i >= 0.5 else 0 for i in y_]
            num = sum(pred_label == y)
            print("accuracy = %s" % str(num / len(X)))

    def predict(self, X):
        """

        :param X:
        :return:
        """
        y = self.sigmoid(np.matmul(X, self.w) + self.intercept)
        return y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred), (1 - y_true) * np.log(1 - y_pred))


if __name__ == '__main__':

    X = np.random.rand(10, 5)
    y = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
    lr = LogisticRegression(lr=0.001, max_iter=100)
    lr.fit(X, y)
    y_ = lr.predict(X)
    print(lr.w)
    print(lr.intercept)
    print(y_)

    print(roc_auc_score(y, y_))
    pred_label = [1 if i >= 0.5 else 0 for i in y_]
    num = sum(pred_label==y)
    print(sum(pred_label==y))
    print("accuracy = %s" % str(num/len(X)))