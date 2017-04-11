
import numpy as np

from .node import Node


class Unitary(Node):

    _op_grad = None

    def __init__(self, operand, code, result_shape=None, prior=0):

        super().__init__(code, prior)

        if result_shape is None:
            result_shape = operand.shape

        self._operand = operand
        self._op_grad = np.zeros(result_shape) if result_shape else 0

        self._shape = result_shape
        self._depth = operand.depth + 1
        self._dependency = (operand,)

    def __repr__(self):
        return self.code + '(' + str(self._operand) + ')'

    def forward(self):
        op_result = self._operand.result
        assert op_result is not None

        if self._op_grad is not None:
            self._gradient = lambda: None
            self._op_grad = None

        self._active = self._operand.active
        self._result = self.eval_op(op_result)

    def backward(self, grad):

        if not self.active:
            self._gradient = lambda: (None,)
            return

        assert grad is not None

        g_shape = grad.shape if self.shape else ()
        assert g_shape == self.shape

        op_shape = self._operand.shape
        assert len(g_shape) == len(op_shape)

        op_result = self._operand.result
        assert op_result is not None

        if not self._op_grad:
            self._gradient = lambda: (self._op_grad,)
            self._op_grad = np.zeros(op_shape) if op_shape else 0

        op_result = self._operand.result
        self._op_grad += grad * self.eval_grad(op_result)

    def eval_op(self, operand):
        raise NotImplementedError

    def eval_grad(self, operand):
        raise NotImplementedError

