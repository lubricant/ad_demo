

import numpy as np

from autodiff.node.binary import Binary
from autodiff.node.unitary import Unitary


class Plus(Binary):

    def __init__(self, a, b):
        super().__init__(a, b, code='+', prior=5)

    def eval_op(self, left, right):
        return left + right

    def eval_grad(self, u, v):
        return 1.0, 1.0


class Minus(Binary):

    def __init__(self, a, b):
        super().__init__(a, b, code='-', prior=5)

    def eval_op(self, left, right):
        return left - right

    def eval_grad(self, u, v):
        return 1.0, -1.0


class Neg(Unitary):
    def __init__(self, o):
        super().__init__(o, code='-', prior=3)

    def __repr__(self):
        o_str, o_prior = str(self._operand), self._operand.prior
        if isinstance(self._operand, Binary) or o_prior > self.prior:
            o_str = '(' + o_str + ')'
        return self.code + o_str

    def eval_op(self, operand):
        return -operand

    def eval_grad(self, operand):
        return -1.0
