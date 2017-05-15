

import numpy as np
import autodiff as ad

from .base import PipelineLayer


class ActiveLayer(PipelineLayer):

    def __init__(self, input_layer, active_type, *args):
        assert isinstance(input_layer, PipelineLayer)
        super(PipelineLayer).__init__('ACT<%s>' % active_type,
                                      input_layer.shape,
                                      input_layer.order)
        active_func = None
        assert active_type in ['sigmoid', 'tanh', 'relu']
        if active_type == 'sigmoid':
            active_func = ad.sigmoid
        if active_type == 'tanh':
            active_func = ad.tanh
        if active_type == 'relu':
            active_func = ad.relu

        self._input = input_layer.output
        self._output = active_func(self._input)

