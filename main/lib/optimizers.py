# coding: utf-8

import numpy as np
import math


class SGD:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, grads):
        for key in weights.keys():
            weights[key] -= self.learning_rate * grads[key]
        return weights


class Momentum:
    def __init__(self, learning_rate = 0.01, alpha = 0.9):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.deltaW = None
        self.name = Momentum

    def update (self, weights, g, mag = 1):
        if self.deltaW == None:
            self.deltaW = {}
            for key, value in weights.items():
                self.deltaW[key] = np.zeros_like(value)

        for key in weights:
            self.deltaW[key] = self.alpha * self.deltaW[key] - mag * self.learning_rate * g[key]
            weights[key] += self.deltaW[key]

        return weights


class AdaGrad:

    def __init__(self, learning_rate = 0.01, eps = 1e-7):
        self.learning_rate = learning_rate
        self.eps = eps
        self.h = None
        self.name = AdaGrad

    def update(self, weights, g, mag = 1):
        if self.h == None:
            self.h = {}
            for key, value in weights.items():
                self.h[key] = np.zeros_like(value)

        for key in weights:
            self.h[key] += g[key] * g[key]
            weights[key] -= mag * self.learning_rate * g[key] / (np.sqrt(self.h[key]) + self.eps)

        return weights

class Adam:

    def __init__(self, input_size, mid_size, out_size, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 10e-8):
        self.t = 0
        self.m = {'W1': np.zeros((input_size, mid_size)), 'b1': np.zeros((mid_size,)),
                  'W2': np.zeros((mid_size, out_size)), 'b2': np.zeros((out_size))}
        self.v = {'W1': np.zeros((input_size, mid_size)), 'b1': np.zeros((mid_size,)),
                  'W2': np.zeros((mid_size, out_size)), 'b2': np.zeros((out_size))}
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def update(self, weights, grads):
        self.t += 1
        for key in weights:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] * grads[key])
            m = self.m[key] / (1 - self.beta1 ** self.t)
            v = self.v[key] / (1 - self.beta2 ** self.t)
            weights[key] -= self.alpha * m / (np.sqrt(v) + self.eps)
        return weights