
import autodiff as ad

from .base import EndpointLayer, PipelineLayer


class InputLayer(EndpointLayer, PipelineLayer):

    def __init__(self, data_shape):
        super().__init__('Input', data_shape)
        self._output = self._input = ad.const(data_shape)

    def feed(self, data):
        self._input.value = data
