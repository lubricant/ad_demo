
import numpy as np
import autodiff as ad


class EndpointLayer(object):

    def feed(self, data):
        raise NotImplementedError


class PipelineLayer(object):

    def __init__(self, name, shape, pre_order=None):
        self._name = name
        self._shape = shape
        self._order = 0 if pre_order is None else pre_order + 1
        self._input = self._output = None

    def __repr__(self):
        return self._name + ('' if not self._shape else str(list(self._shape)))

    @property
    def output(self):
        return self._output

    @property
    def input(self):
        return self._input

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return self._shape


class ParametricLayer(object):

    def param(self):
        raise NotImplementedError

    def grad(self):
        raise NotImplementedError



