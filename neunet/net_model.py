"""
============================

============================
"""

import numpy as np
import autodiff as ad

from neunet import ClassifierModel
from neunet.layer import *


class BinaryNeuralNetwork(ClassifierModel):

    def __init__(self, batch_size):

        assert batch_size > 0
        self._batch_size = batch_size

        input_layer = InputLayer((batch_size, 2))
        hidden_layer1 = FullyConnLayer(input_layer, 6)
        active_layer1 = ActiveLayer(hidden_layer1, ad.tanh)
        hidden_layer2 = FullyConnLayer(active_layer1, 2)
        active_layer2 = ActiveLayer(hidden_layer2, ad.tanh)
        softmax_layer = SoftmaxLayer(active_layer2)

        self._data_layer = input_layer
        self._loss_layer = softmax_layer
        self._param_layers = (hidden_layer1, hidden_layer2)

        score_input_layer = InputLayer((2,))
        score_hidden_layer1 = FullyConnLayer(score_input_layer, 6)
        score_active_layer1 = ActiveLayer(score_hidden_layer1, ad.tanh)
        score_hidden_layer2 = FullyConnLayer(score_active_layer1, 2)
        score_active_layer2 = ActiveLayer(score_hidden_layer2, ad.tanh)

        score_hidden_layer1.replace(hidden_layer1.param())
        score_hidden_layer2.replace(hidden_layer2.param())

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

        # hide1 = self._param_layers[0].param()[1]
        # hide2 = self._score_layers[1].param()[1]
        # print('hide1: ', hide1.value)
        # print('hide2: ', hide2.value)
        # print('----------------------------------------------')

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







