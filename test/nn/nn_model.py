"""
============================

============================
"""

import numpy as np
import autodiff as ad

from .nn_trainer import SGDTrainer


class BinaryClassifierModel(object):

    def update(self, batch):
        print(batch)
        return 0

    def predict(self, x, y):
        return 1 if x == y else 0


class NeuralNetwork(BinaryClassifierModel):

    def __init__(self, layer_num=3, **args):

        assert layer_num >= 2

        self.step = 0.01
        self.weight = []

        self.input = ad.const((2,))
        self.hidden = []
        self.output = None

        x = self.input
        for i in range(layer_num-1):
            w = ad.var('W_%d' % (i+1), np.random.normal(0., 0.0001, (2,2)))
            self.weight.append(w)
            x = ad.sigmoid(x@w)
            self.hidden.append(x)

        w = ad.var('W_%d' % layer_num, np.random.normal(0., 0.0001, (2,)))
        self.weight.append(w)
        # self.output = ad.sigmoid(x@w)
        self.output = ad.softmax(x@w)

        self.expect = ad.const(())
        # self.loss = (self.output - self.expect) ** ad.const(2)
        self.loss = self.output[self.expect]

        print(self.output)
        print(self.loss)

        self.trainer = SGDTrainer(self, **args)

    def __repr__(self):
        return str(self.output)

    def predict(self, x, y):
        self.input.value = np.array([x, y])
        ad.eval(self.output, False)
        return self.output.result > 0.5

    def update(self, batch):
        if not batch:
            return
        return self.trainer.update(batch)


if __name__ == '__main__':
    # nn = NeuralNetwork()
    # print(nn)
    # print(nn.output.result)
    # nn.predict(0,0)
    # print(nn.output.result)

    def check_grad(f, x, eps):
        return (f(x+eps) - f(x-eps))/2/eps
