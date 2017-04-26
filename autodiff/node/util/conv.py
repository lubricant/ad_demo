

from autodiff.node.node import Node
from autodiff.node.binary import Binary
from autodiff.node.unitary import Unitary

import numpy as np


class Convolute(Binary):

    def __init__(self, image, kernel, **args):
        img_shape, k_shape = image.shape, kernel.shape
        assert len(img_shape) >= len(k_shape) > 1

        stride, pad_mode, kernel_dim = 1, 'zero', len(k_shape)

        if 'stride' in args:
            stride = args['stride']
            assert stride

        if 'pad_mode' in args:
            pad_mode = args['pad_mode']
            assert pad_mode in ['zero', 'same']

        if 'kernel_dim' in args:
            kernel_dim = args['kernel_dim']

        super().__init__(image, kernel, code='#', prior=0)
