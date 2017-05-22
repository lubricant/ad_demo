
import numpy as np
import autodiff as ad

from .base import PipelineLayer, ParametricLayer


class FullyConnLayer(PipelineLayer, ParametricLayer):

    def __init__(self, input_layer, neuron_num, rand_seed=None, reuse_param=None, *args):
        assert isinstance(input_layer, PipelineLayer)
        input_order = input_layer.order
        input_shape = input_layer.shape
        hidden_shape = input_shape[-1], neuron_num

        weight_shape = hidden_shape
        bias_shape = (neuron_num,)

        if reuse_param is not None:
            assert len(reuse_param) == 2
            shared_weight, shared_bias = reuse_param
            assert shared_weight is not None and shared_bias is not None
            assert shared_weight.shape == weight_shape and shared_bias.shape == bias_shape
            self._weight, self._bias = reuse_param
        else:
            rand = np.random if rand_seed is None else np.random.RandomState(rand_seed)
            scale = 1. / np.sqrt(sum(weight_shape))
            self._bias = ad.var('FC{bias}', np.zeros(bias_shape))
            self._weight = ad.var('FC{weight}', rand.normal(0.0, scale, weight_shape))

        w, b = self._weight, self._bias
        x = input_layer.output
        y = x @ w + b

        super().__init__('FC', x, y, input_order)

    def param(self):
        return self._weight, self._bias

    def grad(self):
        return self._weight.gradient[0], self._bias.gradient[0]

