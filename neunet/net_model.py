"""
============================

============================
"""

import numpy as np
import autodiff as ad

from neunet import BinaryClassifierModel
from neunet.layer import *


class NeuralNetwork(BinaryClassifierModel):

    def __init__(self, batch_size):
        input_layer = InputLayer((batch_size, 2))
        hidden_layer1 = FullyConnLayer(input_layer, 6)
        active_layer1 = ActiveLayer(hidden_layer1, ad.tanh)
        hidden_layer2 = FullyConnLayer(active_layer1, 2)
        active_layer2 = ActiveLayer(hidden_layer2, ad.tanh)
        softmax_layer = SoftmaxLayer(active_layer2)

        self._param_layers = [hidden_layer1, hidden_layer2]

        self._data_layer = input_layer
        self._loss_layer = softmax_layer
        self._score_layers = [input_layer,
                              hidden_layer1,
                              active_layer1,
                              hidden_layer2,
                              active_layer2]

    def __repr__(self):
        score = ' => '.join([str(l) for l in self._score_layers])
        loss = str(self._loss_layer)
        return 'score: %s \n loss: %s ' % (score, loss)

    def predict(self, x):
        assert isinstance(x, (list, np.ndarray))
        s_in, s_out = self._score_layers[0], self._score_layers[-1]
        s_in.feed(np.array(x)[np.newaxis, :])
        return np.argmax(s_out.eval(), axis=1)

    def eval_data_loss(self, batch_data, batch_label):
        data_placeholder = self._data_layer
        data_placeholder.feed(batch_data)

        loss_func = self._loss_layer
        loss_func.feed(batch_label)

        data_loss = loss_func.eval(need_grad=True)
        return np.mean(data_loss, axis=1)

    def fetch_param_and_grad(self):
        pg_list = []
        for layer in self._param_layers:
            assert isinstance(layer, ParametricLayer)
            param, grad = layer.param(), layer.grad()
            assert len(param) == len(grad)
            pg_list += [x for x in zip(param, grad)]
        return pg_list







