
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

        o_shape = self._operand.shape

        if self.shape == ():
            assert isinstance(grad, (int, float))
            size = o_shape[0]
            self._op_grad += grad / size
        else:
            size = o_shape[self._axis]
            high, _ = self.split_shape(self._axis, o_shape)

            grad = grad / size
            if not len(high):
                self._op_grad += grad
            else:
                high = tuple([slice(0, x) for x in high])
                for i in range(size):
                    self._op_grad[high + (i,)] += grad


class Sum(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'sum', axis)

    def eval_op(self, operand):
        return np.sum(operand, axis=self._axis)

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        o_shape = self._operand.shape

        if self.shape == ():
            assert isinstance(grad, (int, float))
            self._op_grad += grad
        else:
            size = o_shape[self._axis]
            high, _ = self.split_shape(self._axis, o_shape)

            if not len(high):
                self._op_grad += grad
            else:
                high = tuple([slice(0, x) for x in high])
                for i in range(size):
                    self._op_grad[high + (i,)] += grad


class Prod(Reduce):

    def __init__(self, o, axis=None):
        super().__init__(o, 'prod', axis)

    def eval_op(self, operand):
        return np.prod(operand, axis=self._axis)

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        o_shape = self._operand.shape
        o_value = self._operand.result
        o_prod = self.result

        if self.shape == ():
            assert isinstance(grad, (int, float))
            if o_prod:
                o_grad = o_value / o_prod
                o_grad *= grad
                self._op_grad += o_grad
        else:
            size = o_shape[self._axis]
            high, _ = self.split_shape(self._axis, o_shape)

            if not len(high):
                o_grad = np.true_divide(o_value, o_prod)
                o_grad[~np.isfinite(o_grad)] = 0.
                o_grad *= grad
                self._op_grad += o_grad
            else:
                buff = np.zeros(grad.shape)
                high = tuple([slice(0, x) for x in high])
                for i in range(size):
                    np.true_divide(o_value[high + (i,)], o_prod, buff)
                    buff[~np.isfinite(buff)] = 0.
                    buff *= grad
                    self._op_grad[high + (i,)] += buff
