# coding: utf-8

import numpy as np


def sigmoid(x):
    return 1 / 1 + np.exp(-x)


def softmax(x):
    ma = np.max(x)
    e = np.exp(x - ma)
    su = np.sum(np.exp(x - ma))
    return e / su


def cross_entropy(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
