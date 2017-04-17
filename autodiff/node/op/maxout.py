import numpy as np

from autodiff.node.binary import Binary
from autodiff.node.unitary import Unitary


class Maxout(Unitary):

    def __init__(self, o, group_size):
        assert len(o.shape) == 3 and isinstance(group_size, int)
        in_depth, in_row, in_col = o.shape
        out_depth = in_depth // group_size
        assert out_depth > 0
        super().__init__(o, 'maxout', (out_depth, in_row, in_col))

        self._max_idx = None
        self._grp_size = group_size

    def __repr__(self):
        o_name = str(self._operand)
        g_size = str(self._grp_size)
        return self.code + '(' + o_name + ',' + g_size + ')'

    def eval_op(self, operand):
        max_out = np.zeros(self.shape)
        grp_size = self._grp_size

        in_depth, _, _ = operand.shpae
        out_depth, _, _ = self.shape

        max_idx = np.zeros(self.shape)
        for i in range(out_depth):
            beg = i * grp_size
            end = np.minimum(in_depth, beg + grp_size)
            stack = operand[beg:end]
            max_out[i] = stack.max(axis=0)
            max_idx[i] = stack.argmax(axis=0)

        self._max_idx = max_idx
        return max_out

    def backward(self, grad):

        if not self.active:
            self._gradient = lambda: (None,)
            return

        assert grad is not None
        g_shape = grad.shape if self.shape else ()
        assert g_shape == self.shape

        op_shape = self.shape
        assert len(g_shape) == len(op_shape)

        if not self._op_grad:
            g_shape = self._operand.shape
            self._gradient = lambda: (self._op_grad,)
            self._op_grad = np.zeros(g_shape)

        op_grad = self._op_grad
        max_idx = self._max_idx
        assert grad.shape == max_idx.shape

        _, row, col = max_idx.shape
        for i in range(row):
            for j in range(col):
                op_grad[max_idx[:, i, j]] += grad[:, i, j]

