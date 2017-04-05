
import numpy as np

from .node import Node


class Variable(Node):

    def __init__(self, name, value):
        super().__init__(name, 0)

        assert value is not None
        assert isinstance(value, (int, float, tuple, np.ndarray))

        self._depth = 0
        self._val_grad = None

        self._dependency = None
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
        return self.code

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
        if self._val_grad is not None:
            self._gradient = lambda: None
            self._val_grad = None

    def backward(self, grad):
        assert self._result is not None
        assert grad is not None

        g_shape = grad.shape if self.shape else ()
        assert g_shape == self.shape

        if self._val_grad is None:
            self._gradient = lambda: (self._val_grad,)
            self._val_grad = np.zeros(self._shape) if self._shape else 0

        self._val_grad += grad


class Const(Variable):

    def __init__(self, value):
        super(Variable).__init__(str(value), value)

    def backward(self, grad):
        pass






