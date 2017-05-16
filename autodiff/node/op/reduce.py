
from autodiff.node.unitary import Unitary

import numpy as np


class Reduce(Unitary):

    def __init__(self, o, name, axis):

        assert isinstance(axis, (int, None))
        if axis is not None:
            pass

        self._axis = axis
        super().__init__(o, code=name + ('' if axis is None else '[%d]' % axis))


class Mean(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'mean', axis)

    def eval_op(self, operand):
        return np.sin(operand)

    def eval_grad(self, operand):
        return np.cos(operand)


class Sum(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'sum', axis)

    def eval_op(self, operand):
        return np.sin(operand)

    def eval_grad(self, operand):
        return np.cos(operand)


class Prod(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'prod', axis)

    def eval_op(self, operand):
        return np.sin(operand)

    def eval_grad(self, operand):
        return np.cos(operand)
