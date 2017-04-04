
import numpy as np

from autodiff.node.unitary import Unitary


class Slice(Unitary):

    def __init__(self, array, index):

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
                assert not index.step
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

        super(Unitary).__init__(array, guess_shape())
        self.__index = index

    def forward(self):
        op_result = self._operand.result
        assert op_result is not None

        if self._gradient is not None:
            self._gradient = None
            self._op_grad = None

        self._result = op_result[self.__index]

    def backward(self, grad):
        assert grad is not None
        assert grad.shape == self.shape

        op_shape = self.shape
        assert len(grad.shape) == len(op_shape)

        if not self._gradient:
            g_shape = self._operand.shape
            self._op_grad = np.zeros(g_shape)
            self._gradient = (self._op_grad,)

        self._op_grad[self.__index] += grad
