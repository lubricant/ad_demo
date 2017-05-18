
import autodiff as ad

from .base import EndpointLayer, PipelineLayer


class SoftmaxLayer(EndpointLayer, PipelineLayer):

    def __init__(self, input_layer, *args):
        assert isinstance(input_layer, PipelineLayer)
        input_order = input_layer.order
        input_shape = input_layer.shape
        assert len(input_shape) == 2

        batch_size, classes_num = input_shape
        super(PipelineLayer).__init__('SAX', (classes_num,), input_order)

        self._expect = ad.const((batch_size,))
        self._input = input_layer.output
        self._output = ad.softmax(self._input, self._expect)

    def feed(self, data):
        self._expect.value = data

