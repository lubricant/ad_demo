
import numpy as np
import autodiff as ad

from .base import PipelineLayer, ParametricLayer


class FullyConnLayer(PipelineLayer, ParametricLayer):

    def __init__(self, input_layer, neuron_num, rand_seed=None, *args):
        assert isinstance(input_layer, PipelineLayer)
        input_order = input_layer.order
        input_shape = input_layer.shape
        hidden_shape = input_shape[-1], neuron_num

        rand = np.random if rand_seed is None else np.random.RandomState(rand_seed)
        scale = 1. / np.sqrt(sum(hidden_shape))
        self._bias = ad.var('FC{bias}', np.zeros((neuron_num,)))
        self._weight = ad.var('FC{weight}', rand.normal(0.0, scale, hidden_shape))

        w, b = self._weight, self._bias
        x = input_layer.output
        y = x @ w + b

        super().__init__('FC', x, y, input_order)

    def replace(self, param):
        assert len(param) == 2
        new_weight, new_bias = param
        assert new_weight is not None
        assert new_bias is not None
        assert new_weight.shape == self._weight.shape
        assert new_bias.shape == self._bias.shape
        self._weight, self._bias = param

    def param(self):
        return self._weight, self._bias

    def grad(self):
        return self._weight.gradient[0], self._bias.gradient[0]

