# coding: utf-8
import numpy as np
import pytest
import sys
import os
sys.path.append(os.getcwd() + "/../../")
from main.lib import functions as fun


def test_sigmoid_01():
    x = np.array([-1.0, 1.0, 2.0])
    y = np.array([0.26894142, 0.73105858, 0.88079708])
    assert fun.sigmoid(x) == pytest.approx(y)


def test_softmax_01():
    x = np.array([1010, 1000, 990])
    y = np.array([9.99954600e-01, 4.53978686e-05, 2.06106005e-09])
    assert fun.softmax(x) == pytest.approx(y)


def test_softmax_02():
    x = np.array([0.3, 2.9, 4.0])
    y = np.array([0.01821127, 0.24519181, 0.73659691])
    assert fun.softmax(x) == pytest.approx(y)
    