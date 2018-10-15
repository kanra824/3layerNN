# coding: utf-8

import sys
import os

import numpy as np
import functions as fun

class Sigmoid:

    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = fun.sigmoid(x)
        return self.out

    def backward(self, diff):
        return diff * (1.0 - self.out) * self.out


class Affine:

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, diff):
        self.dW = np.dot(self.x.T, diff)
        self.db = np.sum(diff, axis=0)
        dx = np.dot(diff, self.W.T)
        return dx


class SmLo:

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = fun.softmax(x)
        self.loss = fun.cross_entropy(self.y, self.t)

        return self.loss

    def backward(self, diff=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx = dx / batch_size
        return dx

class ReLU:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, diff):
        diff[self.mask] = 0
        dx = diff
        return dx