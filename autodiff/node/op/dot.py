
import numpy as np

from autodiff.node.binary import Binary
from autodiff.node.binary import guess_mat_op_result_shape
from autodiff.node.binary import reduce_grad_shape


class MatMul(Binary):

    def __init__(self, a, b):
        super().__init__(a, b, code='@', prior=4, guess_func=guess_mat_op_result_shape)

    def eval_op(self, left, right):
        return np.matmul(left, right)

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        left, right = self._left, self._right
        l_grad, r_grad = self.eval_mat_grad(grad, left.result, right.result)

        assert l_grad.shape == left.shape
        assert r_grad.shape == right.shape

        self._left_grad += l_grad
        self._right_grad += r_grad

    @staticmethod
    def eval_mat_grad(grad, left, right):
        '''
        给定左右两个操作数 left, right 以及上一层反向传播下来的梯度 grad
        计算两个操作数的反向梯度 g(left) * grad 以及 g(right) * grad
        其中，函数 g(?) 表示 ? 的原始梯度值

        并且最终保证计算结果的形状与操作数相同，即:
            (g(left) * grad).shape = left.shape
            (g(right) * grad).shape = right.shape

        矩阵乘法可以简单地分解为乘法与加法

        首先展示如何计算反向梯度
        令 A、B 分别为左右两边的操作数，可能的情形如下：
            1. 如果两者均为 vector，则计算结果为 scalar：
                A = left = [a,a,a], B = right = [b,b,b]

                令 C = A@B = c 且 g(F) = grad = g

                由于 g(A) = B 因此 g(A).shape = (3,)
                由于 g(B) = A 因此 g(B).shape = (3,)

                根据传递法则有，要求得 ：
                g(A) * g(C) = B * grad = [b0,b1,b2] * g  (3,)
                g(B) * g(C) = A * grad = [a0,a1,a2] * g  (3,)

                由于 g(A) * g(C) 的 shape 与 A 相同，故不用修改梯度形状，可直接用于反向传播
                同上 g(B) * g(C) 亦可直接反向传播

            2. 如果两者均为 matrix，则计算结果为 matrix
                A = left = [[a, a, a, a]    B = right = [[b, b, b]
                            [a, a, a, a]]                [b, b, b]
                                                         [b, b, b]
                                                         [b, b, b]]

                令 C = A@B = [[c, c, c]   =  [[a_0?*b_?0, a_0?*b_?1, a_0?*b_?2]
                              [c, c, c]]      [a_1?*b_?0, a_1?*b_?1, a_1?*b_?2]]

                有 g(A) = B, g(B) = A, g(C) = grad = [[g_00, g_01, g_02]
                                                      [g_10, g_11, g_12]]
                根据传递法则有，要求得：
                g(A) * g(C) = grad * B.T
                g(B) * g(C) = A.T * grad

            3. 如果两者均为 tensor，则计算结果也为 tensor
               这种情况与 3 相似，只需在转置时注意只转置最后两个维度即可

            4. 如果其一为 vector 另一个为其他类型
               此时分为 4 种情况：
                a. vector 在左，matrix 在右，结果为 vector
                    x = left = [a, a, a]  W = right = [[b, b]
                                                       [b, b]
                                                       [b, b]]
                    令 F = x@W = [c, c] = [ a0*b00 + a0*b10 + a0*b20,
                                            a0*b01 + a0*b11 + a0*b21 ]

                    有 g(x) = W，g(W) = x, g(F) = grad = [g, g]
                    根据传递法则有，要求得：

                                              [[b, b]             [[b*g, b*g]
                    g(x) * g(F) = W * grad  =  [b, b]  * [g, g] =  [b*g, b*g]    (3,2)
                                               [b, b]]             [b*g, b*g]]

                                               [[a]               [[a*g, a*g]
                    g(W) * g(F) = x.T * grad =  [a]    * [g, g] =  [a*g, a*g]    (3,2)
                                                [a]]               [a*g, a*g]]

                    由于 g(x) * g(F) 的 shape 与原始操作数 x 的 shape 不一致
                    先对行求和，后转置，将其的形状由 (3,2) 压缩至 (3,)：

                                  [[b_00*g_0, b_01*g_1]    [[b_00*g_0 + b_01*g_1]
                    g(x) * g(F) =  [b_10*g_0, b_11*g_1]  =  [b_10*g_0 + b_11*g_1]   = [b0?*g?, b1?*g?, b2?*g?] (3,)
                                   [b_20*g_0, b_21*g_1]]    [b_20*g_0 + b_21*g_1]]

                -  vector 在左，tensor 在右，结果为 matrix 或 tensor
                   x.shape = (3,)  W.shape = (...,3,2)

                   令 F = x@W  F.shape = (..., 2)
                   有 grad.shape = g(F).shape = F.shape

                   基本与上面类似，需要注意将 grad 提升一个维度
                   即：将 (..., ?) 提升为 (..., 1, ?)

                   g(F).shape = grad.shape = (..., 1, 2)
                   g(x) * g(F) = W * grad = (..., 3, 2) * (..., 1, 2) = (..., 3, 2)
                   g(W) * g(F) = x.T * grad = (3, 1) @ (..., 1, 2) = (..., 3, 2)

                   然后将 g(x) * g(F) 的 shape 通过求和压缩为 (..., 3)
                   然后调用 reduce 将维度降为 (3, )

                b. matrix 在左，vector 在右，算结果为 vector
                    W = left = [[a, a],  x = right = [b, b]
                                [a, a],
                                [a, a]]

                    令 F = W@x = [c,c,c]  = [ a00*b0 + a01*b1,
                                              a10*b0 + a11*b1,
                                              a20*b0 + a21*b1 ]

                    有 g(W) = x, g(x) = W, g(F) = grad = [g,g,g]
                    根据传递法则有，要求得：
                                               [[g]                [[b*g, b*g]
                    g(W) * g(F) = grad.T * x =  [g]    * [b, b]  =  [b*g, b*g]    (3,2)
                                                [g]]                [b*g, b*g]]

                                               [[a, a]    [[g]     [[a*g, a*g]
                    g(x) * g(F) = W * grad.T =  [a, a]  *  [g]   =  [a*g, a*g]    (3,2)
                                                [a, a]]    [g]]     [a*g, a*g]]


                    由于 g(x) * g(F) 的 shape 与原始操作数 x 的 shape 不一致，需要作出修改：
                    先对列求和，后转置，将 g(x) * g(F) 的形状由 (3,2) 压缩至 (2,)

                                  [[a_00*g_0, a_01*g_1]    [[a_00*g_0 + a_10*g_0 + a_20*g_0]
                    g(x) * g(F) =  [a_10*g_0, a_11*g_1]  =  [a_01*g_1 + a_11*g_1 + a_21*g_1]]  = [a?0*g0, a?1*g1] (2,)
                                   [a_20*g_0, a_21*g_1]]


                -  tensor 在左，vector 在右，结果为 matrix 或 tensor
                   W.shape = (...,3,2)  x.shape = (2,)

                   令 F = W@x  F.shape = (..., 3)
                   有 grad.shape = g(F).shape = F.shape

                   基本与上面类似，需要注意将 grad 提升一个维度
                   即：将 (..., ?) 提升为 (..., 1, ?)
                   由于后面计算中使用的是 grad.T，可以直接转置为 (..., ?, 1)

                   g(F).shape = grad.shape = (..., 1, 3)
                   g(W) * g(F) = grad.T * x = (..., 3, 1) @ (2,) = (..., 3, 2)
                   g(x) * g(F) = W * grad.T = (..., 3, 2) * (...,3, 1) = (..., 3, 2)

                   然后将 g(x) * g(F) 的 shape 通过求和压缩为 (..., 2)
                   然后调用 reduce 将维度降为 (2,)


        '''

        l_shape, r_shape = left.shape, right.shape
        g_shape = () if isinstance(grad, (int, float)) else grad.shape

        l_dim, r_dim, g_dim = len(l_shape), len(r_shape), len(g_shape)

        # 情形 1
        if l_dim == r_dim == 1:
            assert g_shape == ()
            return right * grad, left * grad

        # 情形 2
        if l_dim == r_dim == 2:
            assert g_dim == 2
            assert g_shape == (l_shape[-2], r_shape[-1])
            return grad @ right.T, left.T @ grad

        # 情形 3
        if l_dim > 2 and r_dim > 2:
            assert g_dim > 2 and g_dim >= l_dim and g_dim >= r_dim
            assert g_shape[-2:] == (l_shape[-2], r_shape[-1])
            left_t = left.reshape(l_shape[:-2] + l_shape[:-3:-1])
            right_t = right.reshape(r_shape[:-2] + r_shape[:-3:-1])
            l_grad = reduce_grad_shape(grad @ right_t, l_shape)
            r_grad = reduce_grad_shape(left_t @ grad, r_shape)
            return l_grad, r_grad

        # 情形 4
        if l_dim == 1 or r_dim == 1:

            # 情形 a
            if r_dim == 2:
                assert g_shape == (r_shape[-1],)
                l_grad = np.sum(right * grad, axis=-1)
                r_grad = left.reshape(l_shape + (1,)) * grad
                return l_grad, r_grad

            if r_dim > 2:
                assert g_shape[-1] == r_shape[-1]
                grad = grad.reshape(g_shape[:-1] + (1, g_shape[-1]))
                l_grad = reduce_grad_shape(np.sum(right * grad, axis=-1), l_shape)
                r_grad = left.reshape(l_shape + (1,)) @ grad
                return l_grad, r_grad

            # 情形 b
            if l_dim == 2:
                assert g_shape == (l_shape[-2],)
                grad_t = grad.reshape(grad.shape + (1,))
                l_grad = grad_t * right
                r_grad = np.sum(left * grad_t, axis=-2)
                return l_grad, r_grad

            if l_dim > 2:
                assert g_shape[-1] == l_shape[-2]
                grad_t = grad.reshape(g_shape[:-1] + (g_shape[-1], 1))
                l_grad = grad_t @ right.reshape((1,) + r_shape)
                r_grad = reduce_grad_shape(np.sum(left * grad_t, axis=-2), r_shape)
                return l_grad, r_grad

        raise NotImplementedError


if __name__ == "__main__":
    # 情况 2
    A = np.arange(8).reshape((2, 4))
    B = np.arange(12).reshape((4, 3))
    C = A @ B
    print(A)
    print(B)
    print(C)
    print()

    D = np.ones((2, 3))
    print(D @ B.T)
    print(A.T @ D)
    print('---------------------------------')

    # 情况 3
    A = np.arange(2*2*2*3).reshape((2,2,2,3))
    B = np.arange(2*1*3*2).reshape((2,1,3,2))
    C = A @ B
    print(C.shape)
    print(C)

    D = np.ones((2,2,2,2))
    AT = A.reshape((2,2,3,2))
    BT = B.reshape((2,1,2,3))
    GA,GB = D@BT, AT@D
    print(GA.shape)
    print(GA)
    print(GB.shape)
    print(GB)

    GA = reduce_grad_shape(GA,(2,2,2,3))
    GB = reduce_grad_shape(GB,(2,1,3,2))
    print(GA.shape)
    print(GA)
    print(GB.shape)
    print(GB)
    print('---------------------------------')

    # 情况 4
    A = np.array([1,2,3]) # (3,)
    B = np.arange(5*2*3*2).reshape((5,2,3,2))
    C = A@B
    print(C.shape)
    print(C)

    D = np.ones((5, 2, 2))
    GA, GB = MatMul.eval_mat_grad(D, A, B)
    print(GA.shape)
    print(GA)
    print(GB.shape)
    print(GB)
    print('---------------------------------')

    A = np.array([1, 2])  # (2,)
    C = B@A
    print(C.shape)
    print(C)

    D = np.ones((5, 2, 3))
    GB, GA = MatMul.eval_mat_grad(D, B, A)
    print(GA.shape)
    print(GA)
    print(GB.shape)
    print(GB)