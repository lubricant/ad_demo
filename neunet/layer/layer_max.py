

from .base import PipelineLayer


class FullyConnLayer(PipelineLayer):

    def __init__(self, input_layer, *args):
        assert isinstance(input_layer, PipelineLayer)
        input_order = input_layer.order
        input_shape = input_layer.shape
        hidden_shape = input_shape[-1]
        super(PipelineLayer).__init__('SAX', hidden_shape, input_order)