

from autodiff.node.binary import Binary
from autodiff.node.unitary import Unitary

import numpy as np


class Pow(Binary):

    def __init__(self, a, b):
        super(Binary).__init__(a, b)

    def eval_op(self, left, right):
        return left ** right

    def eval_grad(self, u, v):
        uv = (u ** (v-1))
        l_grad = uv * v
        r_grad = uv * u * np.log(u)
        return l_grad, r_grad


class Exp(Unitary):

    def __init__(self, o):
        super(Unitary).__init__(o)

    def eval_op(self, operand):
        return np.exp(operand)

    def eval_grad(self, operand):
        return np.exp(operand)


class Log(Unitary):

    def __init__(self, o):
        super(Unitary).__init__(o)

    def eval_op(self, operand):
        return np.log(operand)

    def eval_grad(self, operand):
        return 1.0 / operand
