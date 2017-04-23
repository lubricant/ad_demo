
import autodiff as ad

from .base import HiddenLayer
from .base import OutputLayer


class FullyConnLayer(HiddenLayer, OutputLayer):

    def __init__(self, input_layer, neuron_num, active_type, *args):
        assert isinstance(input_layer, OutputLayer)
        input_shape = input_layer.output.shape
        output_shape = input_shape[:-1] + (neuron_num,)
        hidden_shape = input_shape[-1], neuron_num

        super(HiddenLayer).__init__('FC<%s>' % active_type, hidden_shape)
        self.bias = None

        x, w = input_layer.output, self.weight

        super(OutputLayer).__init__()