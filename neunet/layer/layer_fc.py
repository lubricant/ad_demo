
import numpy as np
import autodiff as ad

from .base import HiddenLayer
from .base import OutputLayer


class FullyConnLayer(HiddenLayer, OutputLayer):

    def __init__(self, input_layer, neuron_num, *args):
        assert isinstance(input_layer, OutputLayer)
        input_shape = input_layer.output.shape
        hidden_shape = input_shape[-1], neuron_num
        layer_order = input_layer.order if isinstance(input_layer, HiddenLayer) else 0

        scale = np.sqrt(1. / sum(hidden_shape))
        self.bias = ad.var('bias_%d' % layer_order, np.zeros((neuron_num,)))
        self.weight = ad.var('weight_%d' % layer_order, np.random.normal(0.0, scale, hidden_shape))
        x, w, b = input_layer.output, self.weight, self.bias

        super(HiddenLayer).__init__('FC', hidden_shape, layer_order)
        super(OutputLayer).__init__(x @ w + b)

    def grad(self):
        return self.weight[0], self.bias.gradient[0]

    def update(self, value):
        raise NotImplementedError
