
import numpy as np
import autodiff as ad

from .base import PipelineLayer, ParametricLayer


class FullyConnLayer(PipelineLayer, ParametricLayer):

    def __init__(self, input_layer, neuron_num, *args):
        assert isinstance(input_layer, PipelineLayer)
        input_order = input_layer.order
        input_shape = input_layer.shape
        hidden_shape = input_shape[-1], neuron_num
        super(PipelineLayer).__init__('FC', hidden_shape, input_order)

        order = self._order
        scale = 1. / np.sqrt(sum(hidden_shape))
        self.bias = ad.var('FC_%d/bias' % order, np.zeros((neuron_num,)))
        self.weight = ad.var('FC_%d/weight' % order, np.random.normal(0.0, scale, hidden_shape))

        w, b = self.weight, self.bias
        self._input = input_layer.output
        self._output = self._input @ w + b

    def param(self):
        return self.weight, self.bias

    def grad(self):
        return self.weight.gradient, self.bias.gradient

