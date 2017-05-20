

import numpy as np
import autodiff as ad

from .base import PipelineLayer


class ActiveLayer(PipelineLayer):

    def __init__(self, input_layer, active_func, *args):
        assert isinstance(input_layer, PipelineLayer)
        assert active_func in [ad.sigmoid, ad.tanh, ad.relu]

        in_expr = input_layer.output
        out_expr = active_func(in_expr)
        super().__init__('ACT<%s>' % active_func.__name__,
                         in_expr, out_expr,
                         input_layer.order)

