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
        active_layer1 = ActiveLayer(hidden_layer1, 'tanh')
        hidden_layer2 = FullyConnLayer(active_layer1, 2)
        active_layer2 = ActiveLayer(hidden_layer2, 'tanh')
        output_layer = SoftmaxLayer(active_layer2)

        self._feed_layers = [input_layer, output_layer]
        self._param_layers = [hidden_layer1, hidden_layer2]

    def feed_data(self, data, labels):
        head, tail = self._feed_layers
        head.feed(data)
        tail.feed(labels)

    def fetch_param_and_grad(self):
        pg_list = []
        for layer in self._param_layers:
            assert isinstance(layer, ParametricLayer)
            param, grad = layer.param(), layer.grad()
            assert len(param) == len(grad)
            pg_list += [x for x in zip(param, grad)]
        return pg_list







