
import autodiff as ad

from .base import EndpointLayer, PipelineLayer


class SoftmaxLayer(EndpointLayer, PipelineLayer):

    def __init__(self, input_layer, *args):
        assert isinstance(input_layer, PipelineLayer)
        input_order = input_layer.order
        input_shape = input_layer.shape
        assert len(input_shape) == 2

        batch_size, _ = input_shape
        self._expect = ad.const((batch_size,))
        score = input_layer.output
        loss = ad.softmax(score, self._expect)
        super().__init__('SAX', score, loss, input_order)

    def feed(self, data):
        self._expect.value = data

