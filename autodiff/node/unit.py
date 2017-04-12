
import numpy as np

from .node import Node


class Const(Node):

    def __init__(self, value, name=None):

        assert value is not None
        assert isinstance(value, (int, float, tuple, np.ndarray))

        self._depth = 0
        self._val_grad = None

        self._dependency = None
        self._result = None
        self._active = False

        if isinstance(value, tuple):
            self._shape = value
        else:
            if isinstance(value, (int, float)):
                self._shape = ()
            else:
                self._shape = value.shape

            self._result = value

        super().__init__(
            (name if name else '?' if self.result is None else str(value)), 0)

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
        pass


class Variable(Const):

    def __init__(self, name, value):
        super().__init__(value, name)
        self._active = True

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, status):
        assert status is not None
        if not isinstance(status, bool):
            raise ValueError
        self._active = status

    def backward(self, grad):
        assert self._result is not None

        if not self.active:
            self._gradient = lambda: (None,)
            return

        assert grad is not None

        g_shape = grad.shape if self.shape else ()
        assert g_shape == self.shape

        if self._val_grad is None:
            self._gradient = lambda: (self._val_grad,)
            self._val_grad = np.zeros(self._shape) if self._shape else 0

        self._val_grad += grad


