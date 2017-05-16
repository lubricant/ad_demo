
from autodiff.node.unitary import Unitary

import numpy as np


class Reduce(Unitary):

    def __init__(self, o, name, axis):

        assert isinstance(axis, (int, None))

        o_shape = o.shape
        if axis is not None:
            o_shape = o_shape[0:axis] + o_shape[axis:]

        self._axis = axis
        super().__init__(o, code=name + ('' if axis is None else '[%d]' % axis), result_shape=o_shape)


class Mean(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'mean', axis)

    def eval_op(self, operand):
        return np.mean(operand, axis=self._axis)

    def eval_grad(self, operand):
        pass


class Sum(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'sum', axis)

    def eval_op(self, operand):
        return np.sum(operand, axis=self._axis)

    def eval_grad(self, operand):
        pass


class Prod(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'prod', axis)

    def eval_op(self, operand):
        return np.prod(operand, axis=self._axis)

    def eval_grad(self, operand):
        pass
