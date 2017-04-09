
from autodiff.node.unitary import Unitary

import numpy as np


class Tanh(Unitary):

    def __init__(self, o):
        super().__init__(o, code='tanh')

    def eval_op(self, operand):
        return np.tanh(operand)

    def eval_grad(self, operand):
        return 1 - np.tanh(operand) ** 2
