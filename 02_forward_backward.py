#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)

z = 2*x
y = x**2 - z + 1
y.backward(retain_grad=True)
print(z.grad)

