
import numpy as np

from .node import Node


class Variable(Node):

    def __init__(self, name, value):
        super().__init__(name)

        assert value is not None
        assert isinstance(value, (int, float, tuple, np.ndarray))

        self._depth = 0

        self._grad = None

        self._dependency = None
        self._gradient = lambda: self._gradient
        self._result = None

        if isinstance(value, tuple):
            self._shape = value
        else:
            if isinstance(value, (int, float)):
                self._shape = ()
            else:
                self._shape = value.shape

            self._result = value

    def __repr__(self):
        return self.name

    @property
    def value(self):
        return self._result

    @value.setter
    def value(self, value):
        assert value is not None
        if isinstance(value, (int, float)):
            assert self._shape == ()
        elif isinstance(value, np.ndarray):
            assert self._shape == value.shape
        else:
            raise ValueError
        self._result = value

    def forward(self):
        assert self._result is not None
        if self._gradient is not None:
            self._grad = None
            self._gradient = None

    def backward(self, grad):
        assert grad is not None
        assert self._result is not None

        if self._gradient is None:
            self._gradient = (np.zeros(self._shape))


class Const(Variable):

    def __init__(self, value):
        super(Variable).__init__(str(value), value)

    def backward(self, grad):
        pass






