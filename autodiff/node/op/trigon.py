
from autodiff.node.unitary import Unitary

import numpy as np


class Sin(Unitary):

    def __init__(self, o):
        super().__init__(o, code='sin')

    def eval_op(self, operand):
        return np.sin(operand)

    def eval_grad(self, operand):
        return np.cos(operand)


class Cos(Unitary):

    def __init__(self, o):
        super().__init__(o, code='cos')

    def eval_op(self, operand):
        return np.cos(operand)

    def eval_grad(self, operand):
        return -np.sin(operand)
