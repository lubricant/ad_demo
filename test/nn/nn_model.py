"""
============================

============================
"""

import numpy as np
import autodiff as ad


class NeuralNetwork(object):

    def __init__(self, layer_num=2):

        assert layer_num >= 2

        self.step = 0.01
        self.weight = []

        self.input = ad.const((2,))
        self.hidden = []
        self.output = None

        x = self.input
        for i in range(layer_num-1):
            w = ad.var('W_%d' % (i+1), np.random.normal(0., 0.001, (2,2)))
            self.weight.append(w)
            x = ad.sigmoid(x@w)
            self.hidden.append(x)

        w = ad.var('W_%d' % layer_num, np.random.normal(0., 0.001, (2,)))
        self.weight.append(w)
        self.output = ad.sigmoid(x@w)

        self.expect = ad.const(())
        self.loss = (self.output - self.expect) ** ad.const(2)

    def __repr__(self):
        return str(self.output)

    def predict(self, x, y):
        self.input.value = np.array([x, y])
        ad.eval(self.output, False)
        return self.output.result > 0.5

    def update(self, batch):
        if not batch:
            return

        batch_num = len(batch)
        layer_num = len(self.weight)

        loss_sum = 0
        weight_grad_sum = [np.zeros(2,2) for i in range(layer_num)]
        for x, y in batch:
            self.input.value = x
            self.expect.value = y
            loss_sum += ad.eval(self.loss)
            for i in range(layer_num):
                weight_grad_sum[i] += self.weight[i]

        for i in range(layer_num):
            weight_grad = weight_grad_sum[i] / batch_num
            self.weight += -self.step * weight_grad

        return loss_sum / batch_num

if __name__ == '__main__':
    nn = NeuralNetwork()
    print(nn)
    print(nn.output.result)
    nn.predict(0,0)
    print(nn.output.result)
