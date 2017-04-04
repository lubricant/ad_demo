

from autodiff.node.binary import Binary

import numpy as np


class Times(Binary):

    def __init__(self, a, b):
        super(Binary).__init__(a, b)

    def eval_op(self, left, right):
        return left * right

    def eval_grad(self, u, v):
        return v, u


class Divide(Binary):

    def __init__(self, a, b):
        super(Binary).__init__(a, b)

    def eval_op(self, left, right):
        return left / right

    def eval_grad(self, u, v):
        u = self._left.result
        v = self._right.result
        return (1 / v), -(u / (v ** 2))
