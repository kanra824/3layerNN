# coding: utf-8

import sys
sys.path.append("./lib")

import numpy as np
import layers
from collections import OrderedDict


class NeuralNet:

    def __init__(self, input_size, mid_size, out_size, sig=True):
        self.weights = {'W1': np.random.normal(0, 1 / np.sqrt(input_size), (input_size, mid_size)),
                        'b1': np.random.normal(0, 1 / np.sqrt(input_size), (mid_size,)),
                        'W2': np.random.normal(0, 1 / np.sqrt(mid_size), (mid_size, out_size)),
                        'b2': np.random.normal(0, 1 / np.sqrt(mid_size), (out_size,))}

        self.layers = OrderedDict()
        self.layers['Affine1'] = layers.Affine(self.weights['W1'], self.weights['b1'])
        if sig:
            self.layers['Sig'] = layers.Sigmoid()
        else:
            self.layers['ReLU'] = layers.ReLU()
        self.layers['Affine2'] = layers.Affine(self.weights['W2'], self.weights['b2'])
        self.last_layer = layers.SmLo()

    def predict(self, x):
        out = x
        for layer in self.layers.values():
            out = layer.forward(out)
        return out

    def loss(self, x, t):
        result = self.predict(x)
        return self.last_layer.forward(result, t)

    def accuracy(self, x, t):
        result = self.predict(x)
        ans = np.argmax(result, axis=1)
        acc = np.sum(ans == t) / float(x.shape[0])
        return acc

    def grad(self, x, t):
        self.loss(x, t)
        diff = 1
        diff = self.last_layer.backward(diff)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            diff = layer.backward(diff)
        grads = {'W1': self.layers['Affine1'].dW,
                 'b1': self.layers['Affine1'].db,
                 'W2': self.layers['Affine2'].dW,
                 'b2': self.layers['Affine2'].db}
        return grads


if __name__ == '__main__':
    network = NeuralNet(784, 100, 10)
