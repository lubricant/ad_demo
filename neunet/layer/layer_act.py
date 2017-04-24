

import numpy as np
import autodiff as ad

from .base import HiddenLayer
from .base import OutputLayer


class ActiveLayer(HiddenLayer, OutputLayer):

    def __init__(self, input_layer, active_type, *args):
        assert isinstance(input_layer, OutputLayer)
        input_shape = input_layer.output.shape
        hidden_shape = input_shape
        layer_order = input_layer.order if isinstance(input_layer, HiddenLayer) else 0

        x = input_layer.output

        active_func = None
        assert active_type in ['sigmoid', 'tanh', 'relu']
        if active_type == 'sigmoid':
            active_func = ad.sigmoid
        if active_type == 'tanh':
            active_func = ad.tanh
        if active_type == 'relu':
            active_func = ad.relu

        super(HiddenLayer).__init__('ACT<%s>' % active_type, hidden_shape, layer_order)
        super(OutputLayer).__init__(active_func(x))

    def grad(self):
        return ()

    def update(self, value):
        raise NotImplementedError
