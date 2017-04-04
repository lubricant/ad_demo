
import numpy as np

from autodiff.node.binary import Binary


class Max(Binary):

    def __init__(self, a, b):
        super(Binary).__init__(a, b)

    def eval_op(self, left, right):
        return np.maximum(left, right)

    def eval_grad(self, u, v):
        grad_u, grad_v = u-v, v-u
        grad_u[grad_u >= 0] = 1
        grad_u[grad_u < 0] = 0
        grad_v[grad_v > 0] = 1
        grad_v[grad_v <= 0] = 0
        return grad_u, grad_v


class Min(Binary):

    def __init__(self, a, b):
        super(Binary).__init__(a, b)

    def eval_op(self, left, right):
        return np.minimum(left, right)

    def eval_grad(self, u, v):
        grad_u, grad_v = u-v, v-u
        grad_u[grad_u <= 0] = 1
        grad_u[grad_u > 0] = 0
        grad_v[grad_v < 0] = 1
        grad_v[grad_v >= 0] = 0
        return grad_u, grad_v


if __name__ == "__main__":
    a = np.array([1,2,3])
    b = np.array([[0,1,2],[0,2,6]])
    c = np.maximum(a, b)
    print(c)
    ga = a - b
    ga[ga>=0] = 1
    ga[ga<0] = 0
    print(ga)
    print()
    gb = b - a
    gb[gb>0] = 1
    gb[gb<=0] = 0
    print(gb)