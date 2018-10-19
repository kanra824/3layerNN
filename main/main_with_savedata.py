#coding utf-8

import sys
import os
sys.path.append("./lib")
sys.path.append("os.pardir()")
import matplotlib.pyplot as plt
import numpy as np
import pickle
import layers
import optimizers as op
import neuralnet as NN
from mnist import MNIST
mndata = MNIST("../")

input_size = 784
mid_size = 100
out_size = 10
epoch = 1500
batch_size = 100
learning_rate = 0.01
sigmoid = False
optimizers = [op.Adam(784, mid_size, 10)]

print("loading...")
X, Y = mndata.load_training()
print("load training data...")
X_test, Y_test = mndata.load_testing()
print("load testing data...")
if not sigmoid:
    X = np.array(X) / 255
X = X.reshape((X.shape[0], 28 * 28))
Y = np.array(Y)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

for optimizer in optimizers:
    print(optimizer.__class__.__name__)
    with open ('./save/' + optimizer.__class__.__name__, 'rb') as f:
        nn = pickle.load(f)

    acc = nn.accuracy(X_test, Y_test)
    print(acc)