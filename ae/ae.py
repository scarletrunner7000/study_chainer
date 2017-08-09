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

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


iris = load_iris()
xtrain = iris.data.astype(np.float32)

class MyAE(Chain):
    def __init__(self):
        super(MyAE, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 2)
            self.l2 = L.Linear(2, 4)

    def __call__(self, x):
        h1 = self.h1(x)
        loss = F.mean_squared_error(self.l2(h1), x)
        report({'loss': loss}, self)
        return loss

    def h1(self, x):
        return F.sigmoid(self.l1(x))


def plot(model):
    x = Variable(xtrain)
    yt = model.h1(x)
    ans = yt.data
    n = 50
    for i in range(0, len(ans), n):
        plt.scatter(ans[i:(i+n),0], ans[i:(i+n),1])
    plt.savefig('result/encoded2d.eps')


def main():
    model = MyAE()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    epoch = 3000
    batch_size = 30
    train_iter = iterators.SerialIterator(xtrain, batch_size)
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'))

    trainer.extend(extensions.LogReport(), trigger=(100, 'epoch'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    plot(model)


if __name__ == '__main__':
    main()

