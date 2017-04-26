

from autodiff.node.node import Node
from autodiff.node.binary import Binary
from autodiff.node.unitary import Unitary

import numpy as np


class Convolute(Binary):

    def __init__(self, prob, kernel, **args):
        pass