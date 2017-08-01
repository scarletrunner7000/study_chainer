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

f = L.Linear(3, 2)
print(f.W.data)
print(f.b.data)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x) # 普通の関数みたいに書ける
print(y.data)

f.cleargrads() # 蓄積された grad を一旦 clear

y.grad = np.ones((2, 2), dtype=np.float32)
y.backward()
print(f.W.grad)
print(f.b.grad)


