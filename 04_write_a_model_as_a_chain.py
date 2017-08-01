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


class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 3)
            self.l2 = L.Linear(3, 2)

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)


# link の数が静的でない場合は、ChainList class を使えばいい
class MyChain2(ChainList):
    def __init__(self):
        super(MyChain2, self).__init__(
            L.Linear(4, 3),
            L.Linear(3, 2),
        )

    def __call__(self, x):
        h = self[0](x)
        return self[1](h)


