
import numpy as np

from autodiff.node.node import Node
from autodiff.node.unitary import Unitary
from autodiff.node.binary import Binary


class Slice(Unitary):

    def __init__(self, array, index):

        # 根据传入的索引估计根据索引的名称
        # 分成 3 种可能的索引值进行处理
        def guess_code():

            if isinstance(index, int):
                return '[' + str(index) + ']'

            if isinstance(index, slice):
                return '[' + str(index.start) + ':' + str(index.stop) + ']'

            if isinstance(index, tuple):
                idx_list = []
                for idx in index:
                    if isinstance(idx, int):
                        idx_list.append(str(idx))
                    else:
                        idx_list.append(idx.start + ':' + idx.stop)
                return '[' + ','.join(idx_list) + ']'

        # 根据传入的索引估计根据索引取值的形状
        # 分成 3 种可能的索引值进行处理
        def guess_shape():
            arr_shape = array.shape
            assert len(arr_shape) > 0

            if isinstance(index, int):
                assert 0 <= index < arr_shape[0]
                is_vec = len(arr_shape) is 2 and min(arr_shape) is 1
                return () if is_vec else arr_shape[1:]

            if isinstance(index, slice):
                assert not index.step  # 不支持自定义步长
                beg = index.start if index.start else 0
                end = index.stop if index.stop else arr_shape[0]
                assert 0 < (end-beg) < arr_shape[0]
                return (end-beg,) + arr_shape[1:]

            if isinstance(index, tuple):

                arr_dim, idx_dim = len(arr_shape), len(index)
                assert arr_dim >= idx_dim

                # 根据数组某个维度下的长度以及索引值
                # 计算索引取值后，该维度的长度
                def guess_dim(arr_len, idx):
                    if isinstance(idx, int):
                        assert 0 < idx < arr_len
                        return 1
                    if isinstance(idx, slice):
                        assert not idx.step
                        beg = idx.start if idx.start else 0
                        end = idx.stop if idx.stop else arr_len
                        assert 0 < (end-beg) < arr_len
                        return end - beg
                    raise IndexError

                prefix, suffix = arr_shape[:idx_dim], arr_shape[idx_dim:]
                prefix = [guess_dim(prefix[i], index[i]) for i in range(idx_dim)]

                # 如果索引类型为 int，则对结果进行降维
                for idx in index:
                    if isinstance(idx, int):
                        prefix = prefix[1:]
                    else:
                        prefix = tuple(prefix)
                        break

                return prefix + suffix

        super().__init__(array, guess_code(), guess_shape(), prior=1)
        self.__index = index

    def __repr__(self):
        arr_name = str(self._operand)
        if self._operand.prior > self.prior:
            arr_name = '(' + arr_name + ')'
        return arr_name + self.code

    def eval_op(self, operand):
        return operand[self.__index]

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        assert grad is not None

        g_shape = grad.shape if self.shape else ()
        assert len(g_shape) == len(self.shape)

        self._op_grad[self.__index] += grad


class SliceX(Binary):

    def __init__(self, array, index):

        # 只支持整数索引
        assert isinstance(index, Node) and index.shape == ()

        def guess_shape(a, b):
            arr_shape = array.shape
            assert len(arr_shape) > 0
            is_vec = len(arr_shape) is 2 and min(arr_shape) is 1
            return () if is_vec else arr_shape[1:]

        super().__init__(array, index, code='[' + str(index) + ']', prior=1, guess_func=guess_shape)

    def __repr__(self):
        arr_name = str(self._left)
        if self._left.prior > self.prior:
            arr_name = '(' + arr_name + ')'
        return arr_name + self.code

    def eval_op(self, left, right):
        return left[right]

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        assert grad is not None

        g_shape = grad.shape if self.shape else ()
        assert len(g_shape) == len(self.shape)

        index = self._right.result
        assert isinstance(index, int)
        assert 0 <= index < self._left.shape[0]

        self._left_grad[index] += grad
        # ignore right side grad

