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
from pylab import cm
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
print("load training data...")
X, Y = mndata.load_training()
print("load testing data...")
X_test, Y_test = mndata.load_testing()

print("> ", end="")
i = int(input())
X = np.array(X)
Y = np.array(Y)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
Xshow = X_test.reshape((X_test.shape[0], 28, 28))
plt.imshow(Xshow[i], cmap=cm.gray)

if not sigmoid:
    X_test = X_test / 255
X_test = X_test.reshape((X_test.shape[0], 28 * 28))

# for optimizer in optimizers:
#     print(optimizer.__class__.__name__)
#     with open ('./save/' + optimizer.__class__.__name__, 'rb') as f:
#         nn = pickle.load(f)
#
#     acc = nn.accuracy(X_test, Y_test)
#     print(acc)

for optimizer in optimizers:
    with open('./save/' + optimizer.__class__.__name__, 'rb') as f:
        nn = pickle.load(f)
    mask = [i];
    X_batch = X_test[mask]
    out = nn.predict(X_batch)
print(np.argmax(out, axis=1)[0])
plt.show()