

import autodiff.node

import numpy as np


def var(name, val):
    val = __ensure_not_list(val)
    return node.Variable(name, val)


def const(val, name=None):
    val = __ensure_not_list(val)
    return node.Const(val, name)


def sin(x):
    x = __ensure_node(x)
    return node.Sin(x)


def cos(x):
    x = __ensure_node(x)
    return node.Cos(x)


def max(a, b):
    a, b = __ensure_node((a, b))
    return node.Max(a, b)


def min(a, b):
    a, b = __ensure_node((a, b))
    return node.Min(a, b)


def log(x):
    x = __ensure_node(x)
    return node.Log(x)


def exp(x):
    x = __ensure_node(x)
    return node.Exp(x)


def tanh(x):
    x = __ensure_node(x)
    return node.Tanh(x)


def sigmoid(x):
    x = __ensure_node(x)
    return node.Sigmoid(x)


def relu(x):
    x = __ensure_node(x)
    return node.ReLu(x)


def maxout(x, i):
    x = __ensure_node(x)
    return node.Maxout(x, i)


def softmax(x):
    x = __ensure_node(x)
    return node.Softmax(x)


def eval(x, need_grad=True):
    from .chain import Chain
    Chain(x).propagate(need_grad)


def __ensure_node(value):

    if isinstance(value, tuple):
        value = tuple([__ensure_node(v) for v in value])
    elif not isinstance(value, node.Node):
        value = const(value)

    return value


def __ensure_not_list(value):
    if isinstance(value, list):
        value = np.array(value)
    return value
