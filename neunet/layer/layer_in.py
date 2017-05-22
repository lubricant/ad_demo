
import autodiff as ad

from .base import EndpointLayer, PipelineLayer


class InputLayer(PipelineLayer, EndpointLayer):

    def __init__(self, data_shape, use_var=False):
        self._slot = ad.const(data_shape) if not use_var else ad.var('???', data_shape)
        super().__init__('IN', self._slot, self._slot)

    def feed(self, data):
        self._slot.value = data
