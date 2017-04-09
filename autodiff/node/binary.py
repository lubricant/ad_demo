
import numpy as np

from .node import Node


class Binary(Node):

    _left_grad, _right_grad = None, None

    def __init__(self, left, right, code, prior, guess_func=None):

        super().__init__(code, prior)

        self._left, self._right = left, right

        if not guess_func:
            guess_func = guess_op_result_shape

        self._shape = guess_func(left.shape, right.shape)
        self._depth = np.maximum(left.depth, right.depth) + 1
        self._dependency = (self._left, self._right)

    def __repr__(self):
        l_str, r_str = str(self._left), str(self._right)
        if self.prior is 0:
            return self.code + '(' + l_str + ',' + r_str + ')'
        else:
            l_prior, r_prior = self._left.prior, self._right.prior
            if l_prior > self.prior:
                l_str = '(' + l_str + ')'
            if r_prior > self.prior:
                r_str = '(' + r_str + ')'
            return l_str + self.code + r_str

    def forward(self):
        left, right = self._left, self._right
        assert left.result is not None
        assert right.result is not None

        if self._left_grad is not None and self._right_grad is not None:
            self._gradient = lambda: None
            self._left_grad, self._right_grad = None, None

        self._result = self.eval_op(left.result, right.result)

    def backward(self, grad):
        assert grad is not None

        g_shape = grad.shape if self.shape else ()
        assert g_shape == self.shape

        l_shape = self._left.shape
        r_shape = self._right.shape

        assert len(g_shape) >= len(l_shape)
        assert len(g_shape) >= len(r_shape)

        if self._left_grad is None and self._right_grad is None:
            self._gradient = lambda: (self._left_grad, self._right_grad)
            self._left_grad = 0 if l_shape == () else np.zeros(l_shape)
            self._right_grad = 0 if r_shape == () else np.zeros(r_shape)

        left, right = self._left, self._right
        assert left.result is not None
        assert right.result is not None
        l_grad, r_grad = self.eval_grad(left.result, right.result)

        l_grad, r_grad = l_grad * grad, r_grad * grad
        l_grad = reduce_grad_shape(l_grad, l_shape)
        r_grad = reduce_grad_shape(r_grad, r_shape)

        self._left_grad += l_grad
        self._right_grad += r_grad

    def eval_op(self, left, right):
        raise NotImplementedError

    def eval_grad(self, left, right):
        raise NotImplementedError


def reduce_grad_shape(grad, op_shape):
    '''
    当梯度维度高于操作数的维度时，根据 numpy 传播规则对梯度进行降维，下面举例进行分析：
    相对于给定某的个低维度的形状 (a, 1, 1, d)，
    可以将形状为 (..., a, b, 1, c) 的梯度数组划分为如下两部分
        高于 (a, 1, 1, d) 的前缀：(...)
        对齐 (a, 1, 1, d) 的后缀：(a,b,1,c)
    下面分为两步进行处理：
        1. 使用 np.einsum 对 grad 前缀部分进行降维（求和），使其 op_shape 对齐
        2. 使用 np.sum 对后缀部分比 op_shape 大的维度进行降维（求和）
    对求和后的 grad 进行 reshape 令其与 op_shape 一致后返回
    '''

    g_shape = () if isinstance(grad, (int, float)) else grad.shape

    if g_shape == op_shape:
        return grad

    g_dim, op_dim = len(g_shape), len(op_shape)
    assert g_dim >= op_dim

    # 步骤 1
    if g_dim > op_dim:
        exceed, align = tuple(range(g_dim - op_dim)), (Ellipsis,)
        grad = np.einsum(grad, exceed + align, align)
        g_shape = grad.shape
        assert len(g_shape) == op_dim

    # 步骤 2
    offset = 0  # 降维后的轴会向前移动，因此要记录偏移
    for axis in range(op_dim):
        g_size, op_size = g_shape[axis], op_shape[axis]
        if g_size == op_size:
            continue
        assert op_size is 1
        grad = np.sum(grad, axis=axis-offset)
        offset += 1

    return grad.reshape(op_shape)


def guess_op_result_shape(a_shape, b_shape):
    '''
    给定两个数组的维度，计算其运算后的形状，适用于 + - * / **
    参数均为元组，不同情况下的取值可能有：
        a. scalar ()
        b. vector (?,)
        c. matrix (?,?)
        d. tensor (?,...,?)
    此处考虑 numpy 可传播的特性，并进行了一些更为严格的限制，规则如下：
        1. 当 a、b 其中一个是 scalar 时，直接返回另一个数组的形状（标量传播时默认会对齐）
        2. 当 a、b 其中一个是 vector 时：
            - 若两者的最后一维相同，则返回另一个数组的形状
            - 若两者的最后一维不同，则产生异常
        3. 当 a、b 均为更高维的数组时，用 a_shape、b_shape 代表两者的形状：
            - 若 a_shape 包含 b_shape 或者 b_shape 包含 a_shape，则返回两者中较长的形状
              包含的定义为: (..., a, b, c) 包含 （a, b, c）、（b, c）、(c,)
              其他情况以此类推
            - 若 a_shape 与 b_shape 互不包含，则使用 1 将两者填充为同样的维度
              然后判断在是否所有维度上满足:
                首先，在所有不相等的维度上，两者中必须有一个为 1
                其次，在所有不相等的维度上，其中一方都必须大于等于另一方
              如果满足上述条件，则返回维度较高或者维度总数较大者
              如果不满足上述条件，则产生异常
    注意：规则 2 实际上包含在规则 3 之中，然而由于其处理步骤较简单，故独立出来
    '''

    a_dim, b_dim = len(a_shape), len(b_shape)

    # 两者形状相同，运算结果的形状不变
    if a_shape == b_shape:
        return a_shape

    # 规则 1
    if not a_dim or not b_dim:
        return a_shape if not b_dim else b_shape

    # 规则 2
    if a_dim == 1 or b_dim == 1:

        if a_shape[-1] != b_shape[-1]:
            raise ShapeError(a_shape, b_shape)

        return a_shape if (a_dim > b_dim) else b_shape

    # 规则 3-1
    if a_dim > b_dim:
        if a_shape[a_dim-b_dim:] == b_shape:
            return a_shape
    else:
        if b_shape[b_dim-a_dim:] == a_shape:
            return b_shape

    # 规则 3-2
    a_align = ((1,) * max(0, b_dim-a_dim)) + a_shape
    b_align = ((1,) * max(0, a_dim-b_dim)) + b_shape

    ab_cnt, ba_cnt = 0, 0
    for a_size, b_size in zip(a_align, b_align):
        if a_size == b_size:
            continue
        if a_size != 1 and b_size != 1:
            raise ShapeError(a_shape, b_shape)
        ab_cnt += 1 if a_size > b_size else 0
        ba_cnt += 1 if b_size > a_size else 0

    if ab_cnt and ba_cnt:
        raise ShapeError(a_shape, b_shape)

    if a_dim == b_dim:
        return a_shape if ab_cnt > ba_cnt else b_shape
    else:
        return a_shape if a_dim > b_dim else b_shape


def guess_mat_op_result_shape(a_shape, b_shape):
    '''
    给定两个数组的维度，计算其运算后的形状，适用于 @
    可能取值参考上面
    此处考虑 numpy 可传播的特性，并进行了一些更为严格的限制，规则如下：
    1. 当 a、b 其中一个是 scalar 时，产生异常
    2. 当 a、b 均为 vector 时，作为点乘处理：
        - 若 a、b 长度不相等，则产生异常
        - 若 a、b 长度相等，则返回 () 作为结果
    3. 当 a、b 其中一个是 vector 时，使用以下规则对齐，进行下一步处理：
        - 若 vector 在左边，则将其作为列向量 (1,?)
        - 若 vector 在右边，则将其作为行向量 (?,1)
    4. 当 a、b 均为 matrix 时，则按照矩阵乘法法则进行处理，用 row, col 代表两者的行列数：
        - 若 a_col != b_row ，则产生异常
        - 若 a_col == b_row ，则返回 (a_row, b_col) 作为结果
    5. 当 a、b 其中一个是 tensor 时，则将其作为一系列的矩阵进行处理：
        - 首先，比较两者最低两个维度，判断是否满足矩阵乘法的要求
          若不满足计算要求，则产生异常
          若满足计算要求，则按照如下方式获取计算结果：
            - 若两者中不存在向量，则直接按照规则 4 返回结果
            - 若两者中存在向量，则按照规则 4 计算后，对结果进行降维
        - 然后，使用 guess_op_result_shape 计算更高维度的结果
        - 最终，将两部分结果合并，得到最终的形状
    '''

    a_dim, b_dim = len(a_shape), len(b_shape)

    # 规则 1
    if not a_dim or not b_dim:
        raise ShapeError(a_shape, b_shape, True)

    # 规则 2
    if a_dim == b_dim == 1:
        a_size, b_size = a_shape[0], b_shape[0]
        if a_size != b_size:
            raise ShapeError(a_shape, b_shape, True)
        return ()

    # 规则 3
    a_mat = ((1,) if a_dim == 1 else ()) + a_shape
    b_mat = ((1,) if b_dim == 1 else ()) + b_shape

    # 规则 4
    a_row, a_col = a_mat[-2:]
    b_row, b_col = b_mat[-2:]

    if a_col != b_row:
        raise ShapeError(a_shape, b_shape, True)

    if len(a_mat) == len(b_mat) == 2:
        return a_row, b_col

    # 规则 5
    a_row = () if a_dim == 1 else (a_row,)
    b_col = () if b_dim == 1 else (b_col,)
    return guess_op_result_shape(a_mat[:-2], b_mat[:-2]) + a_row + b_col


class ShapeError(ValueError):
    def __init__(self, a_shape, b_shape, mat_op=False):
        super(Exception).__init__(
            ('shape %s and %s are not compatible'
             '' if not mat_op else 'for mat op'
             ) % (a_shape, b_shape))

