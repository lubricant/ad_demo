"""
============================

============================
"""

import numpy as np
import autodiff as ad

from neunet import BinaryClassifierModel


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

        # w = ad.var('W_%d' % layer_num, np.random.normal(0., 0.0001, (2,)))
        w = ad.var('W_%d' % layer_num, np.random.normal(0., 0.0001, (2,2)))
        self.weight.append(w)
        # self.output = ad.sigmoid(x@w)
        self.output = ad.softmax(x@w)

        self.expect = ad.const(())
        # self.loss = (self.output - self.expect) ** ad.const(2)
        self.loss = -ad.log(self.output[self.expect])

        print(self.output)
        print(self.loss)

    def __repr__(self):
        s = 'output: ' + str(self.output) + '\n'
        s += 'loss: ' + str(self.loss)
        return s

    def predict(self, x, y):
        self.input.value = np.array([x, y])
        ad.eval(self.output, False)
        # return self.output.result > 0.5
        return np.argmax(self.output.result)


if __name__ == '__main__':
    # nn = NeuralNetwork()
    # print(nn)
    # print(nn.output.result)
    # nn.predict(0,0)
    # print(nn.output.result)

    def check_grad(f, x, eps):
        return (f(x+eps) - f(x-eps))/2/eps
