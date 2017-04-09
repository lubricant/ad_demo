
from autodiff.node.unitary import Unitary

import numpy as np


class Sigmoid(Unitary):

    def __init__(self, o):
        super().__init__(o, code='sigmoid')

    def eval_op(self, operand):
        return 1 / (1 + np.exp(-operand))

    def eval_grad(self, operand):
        s = 1 / (1 + np.exp(-operand))
        return s * (1 - s)
