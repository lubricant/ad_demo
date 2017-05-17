
from autodiff.node.unitary import Unitary

import numpy as np


class Reduce(Unitary):

    def __init__(self, o, name, axis):

        assert len(o.shape) > 0
        assert axis is None or isinstance(axis, int)

        o_shape = ()
        if axis is not None:
            high, low = self.split_shape(axis, o.shape)
            o_shape = high + low

        self._axis = axis
        super().__init__(o, code=name + ('' if axis is None else '[%d]' % axis), result_shape=o_shape)

    @staticmethod
    def split_shape(axis, shape):
        if axis >= 0:
            return shape[0:axis], shape[axis+1:]
        else:
            return shape[0:axis], shape[axis:-1]

class Mean(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'mean', axis)

    def eval_op(self, operand):
        return np.mean(operand, axis=self._axis)

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        operand = self._operand
        o_shape = operand.shape

        if self.shape == ():
            assert isinstance(grad, (int, float))
            size = operand.shape[0]
            self._op_grad += grad / size
        else:
            axis = self._axis
            size = o_shape[axis]
            high, _ = self.split_shape(axis, o_shape)

            if not len(high):
                self._op_grad += grad / size
            else:
                grad = grad / size
                high = tuple([slice(0, x) for x in high])
                for i in range(axis):
                    print('size:', self._op_grad[high + (i,)].shape)
                    self._op_grad[high + (i,)] += grad


class Sum(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'sum', axis)

    def eval_op(self, operand):
        return np.sum(operand, axis=self._axis)

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        if self.shape == ():
            assert isinstance(grad, (int, float))
            self._op_grad += grad


class Prod(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'prod', axis)

    def eval_op(self, operand):
        return np.prod(operand, axis=self._axis)

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        operand = self._operand
        op_val, = operand.result
        prod_val, = self.result

        if self.shape == ():
            assert isinstance(grad, (int, float))
            if prod_val:
                op_grad = op_val / prod_val
                op_grad *= grad
                self._op_grad += op_grad
