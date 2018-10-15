#coding: utf-8

import sys
import os
sys.path.append("./lib")
sys.path.append("os.pardir()")
import matplotlib.pyplot as plt
import numpy as np
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
optimizers = [op.SGD(),
              op.Momentum(),
              op.AdaGrad(),
              op.Adam(784, mid_size, 10)]


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
    loss_list = []
    nn = NN.NeuralNet(input_size, mid_size, out_size, sigmoid)

    for i in range(epoch):
        mask = np.random.choice(range(0, X.shape[0]), batch_size, replace=False)
        X_batch = X[mask]
        Y_batch = Y[mask]
        grads = nn.grad(X_batch, Y_batch)

        nn.weights = optimizer.update(nn.weights, grads)

        loss = nn.loss(X_batch, Y_batch)
        if i % 100 == 0:
            print(str(i)+ ":" + str(loss))
        if i % 10 == 0:
            loss_list.append(loss)

    acc = nn.accuracy(X_test, Y_test)
    print(acc)
    plt.plot(loss_list, label=optimizer.__class__.__name__ + ":" + str(acc))
plt.title('NeuralNetwork')
plt.xlabel('epoch / 10')
plt.ylabel('loss')

plt.legend()
plt.show()