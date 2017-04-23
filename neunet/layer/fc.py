

from neunet.layer.base import HiddenLayer
from neunet.net_model import OutputLayer


class FullyConnLayer(HiddenLayer, OutputLayer):

    def __init__(self, input_layer, output_shape, active_type):
        super(HiddenLayer).__init__('FullConn', shape)
        super(OutputLayer).__init__()