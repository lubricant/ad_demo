
import numpy as np
import autodiff as ad

from autodiff.node import Node


class OutputLayer(object):

    def __init__(self, output_expr):
        assert isinstance(output_expr, Node)
        self.output = output_expr

    def output(self):
        return self.output


class InputLayer(OutputLayer):

    def __init__(self, input_shape):
        assert len(input_shape) > 0
        self.input = ad.const(input_shape, '???')
        super().__init__(self.input)

    def __repr__(self):
        return 'Input' + str(list(self.input.shape))

    def input(self):
        return self.input


class HiddenLayer(object):

    def __init__(self, order, name, shape, has_weight=True):
        '''
        对权重进行正则化，保证每个神经元输出方差一致
        否则那些有着更多输入的神经元，输出有着更高的方差（过拟合）
        '''
        self.name = name
        self.shape = shape
        if has_weight:
            scale = np.sqrt(1. / sum(shape))
            self.weight = ad.var('%s_%s' % (name, order),
                                 np.random.normal(0.0, scale, shape))

    def __repr__(self):
        return self.name + str(list(self.shape))

    def grad(self):
        if self.weight is not None:
            return self.weight.gradient[0]

    def value(self):
        if self.weight is not None:
            return self.weight.result

    def update(self, value):
        if self.weight is not None:
            self.weight.value = value
