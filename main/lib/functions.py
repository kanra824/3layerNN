# coding: utf-8

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def softmax(x):
#     ma = np.max(x)
#     e = np.exp(x - ma)
#     su = np.sum(np.exp(x - ma))
#     return e / su

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy(y, t):
    eps = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + eps)) / batch_size
