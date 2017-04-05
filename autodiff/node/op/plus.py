

import numpy as np

from autodiff.node.binary import Binary


class Plus(Binary):

    def __init__(self, a, b):
        super().__init__('+', a, b)

    def eval_op(self, left, right):
        return left + right

    def eval_grad(self, u, v):
        return 1.0, 1.0


class Minus(Binary):

    def __init__(self, a, b):
        super().__init__('-', a, b)

    def eval_op(self, left, right):
        return left - right

    def eval_grad(self, u, v):
        return 1.0, -1.0
