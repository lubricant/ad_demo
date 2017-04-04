
import numpy as np

from .node import Node


class Unitary(Node):

    _op_grad = None

    def __init__(self, operand, result_shape=None):

        if result_shape is None:
            result_shape = operand.shape

        self._operand = operand
        self._op_grad = np.zeros(result_shape)

        self._shape = result_shape
        self._depth = operand.depth + 1
        self._dependency = (operand,)
        self._gradient = (self._op_grad,)

    def forward(self):
        op_result = self._operand.result
        assert op_result is not None

        if self._gradient is not None:
            self._gradient = None
            self._op_grad = None

        self._result = self.eval_op(op_result)

    def backward(self, grad):
        assert grad is not None
        assert grad.shape == self.shape

        op_shape = self._operand.shape
        assert len(grad.shape) == len(op_shape)

        op_result = self._operand.result
        assert op_result is not None

        if not self._gradient:
            self._op_grad = np.zeros(op_shape)
            self._gradient = (self._op_grad,)

        op_result = self._operand.result
        self._op_grad += grad * self.eval_grad(op_result)

    def eval_op(self, operand):
        raise NotImplementedError

    def eval_grad(self, operand):
        raise NotImplementedError
