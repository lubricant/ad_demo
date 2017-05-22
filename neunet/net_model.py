"""
============================

============================
"""

import numpy as np
import autodiff as ad

from neunet import ClassifierModel
from neunet.layer import *


class BinaryNeuralNetwork(ClassifierModel):

    def __init__(self, batch_size, rand_seed=None):

        assert batch_size > 0
        self._batch_size = batch_size

        input_layer = InputLayer((batch_size, 2))
        hidden_layer1 = FullyConnLayer(input_layer, 6, rand_seed)
        active_layer1 = ActiveLayer(hidden_layer1, ad.tanh)
        hidden_layer2 = FullyConnLayer(active_layer1, 2, rand_seed)
        active_layer2 = ActiveLayer(hidden_layer2, ad.tanh)
        softmax_layer = SoftmaxLayer(active_layer2)

        self._data_layer = input_layer
        self._loss_layer = softmax_layer
        self._param_layers = (hidden_layer1, hidden_layer2)

        score_input_layer = InputLayer((2,))
        score_hidden_layer1 = FullyConnLayer(score_input_layer, 6, reuse_param=hidden_layer1.param())
        score_active_layer1 = ActiveLayer(score_hidden_layer1, ad.tanh)
        score_hidden_layer2 = FullyConnLayer(score_active_layer1, 2, reuse_param=hidden_layer2.param())
        score_active_layer2 = ActiveLayer(score_hidden_layer2, ad.tanh)

        self._score_io = (score_input_layer, score_active_layer2)
        self._score_layers = (score_input_layer,
                              score_hidden_layer1,
                              score_active_layer1,
                              score_hidden_layer2,
                              score_active_layer2)

    def __repr__(self):
        score = ' => '.join([str(l) for l in self._score_layers])
        loss = str(self._loss_layer)
        return 'score: %s \n loss: %s ' % (score, loss)

    def eval_score(self, x):
        assert isinstance(x, (list, np.ndarray))
        s_in, s_out = self._score_io
        s_in.feed(np.array(x))
        return np.argmax(s_out.eval())

    def eval_loss(self, batch_data, batch_label):
        data_placeholder = self._data_layer
        data_placeholder.feed(batch_data)

        loss_func = self._loss_layer
        loss_func.feed(batch_label)

        data_loss = loss_func.eval(need_grad=True)
        return np.mean(data_loss)

    def list_param_and_grad(self):
        pg_list = []
        for layer in self._param_layers:
            assert isinstance(layer, ParametricLayer)
            param, grad = layer.param(), layer.grad()
            assert len(param) == len(grad)
            pg_list += [x for x in zip(param, grad)]

        return pg_list


if __name__ == '__main__':
    a = InputLayer((1, 2), True)
    b = FullyConnLayer(a, 6, rand_seed=0)
    c = ActiveLayer(b, ad.tanh)
    d = FullyConnLayer(c, 2, rand_seed=0)
    e = ActiveLayer(d, ad.tanh)
    f = SoftmaxLayer(e)

    x, y = np.array([[2., 3.]]), np.array([1])
    a.feed(x)
    f.feed(y)
    f.eval(True)
    x_g = a.input.gradient
    print(x_g)

    # gradient checking

    h = 10.e-5

    # evaluate numeric gradient of x0
    x0_lo, x0_hi = np.array([[2-h, 3]]), np.array([[2+h, 3]])

    a.feed(x0_lo)
    # f.feed(y)
    f0_lo = f.eval()

    a.feed(x0_hi)
    # f.feed(y)
    f0_hi = f.eval()

    x0_g = (f0_hi - f0_lo) / (2 * h)

    # evaluate numeric gradient of x1
    x1_lo, x1_hi = np.array([[2, 3- h]]), np.array([[2, 3 + h]])

    a.feed(x1_lo)
    # f.feed(y)
    f1_lo = f.eval()

    a.feed(x1_hi)
    # f.feed(y)
    f1_hi = f.eval()

    x1_g = (f1_hi - f1_lo) / (2 * h)

    # combine gradient
    print(np.hstack((x0_g, x1_g)))





