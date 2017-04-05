
class Node(object):

    _depth = -1
    _shape = ()

    __code = None
    __prior = -1

    _result = None
    _dependency = ()

    shape = property(lambda self: self._shape, lambda self, v: None, lambda self: None)
    depth = property(lambda self: self._depth, lambda self, v: None, lambda self: None)

    code = property(lambda self: self.__code, lambda self, v: None, lambda self: None)
    prior = property(lambda self: self.__prior, lambda self, v: None, lambda self: None)

    result = property(lambda self: self._result, lambda self, v: None, lambda self: None)
    gradient = property(lambda self: self._gradient(), lambda self, v: None, lambda self: None)
    dependency = property(lambda self: self._dependency, lambda self, v: None, lambda self: None)

    def __add__(self, o):
        assert o is not None
        assert isinstance(o, Node)
        from . import Plus
        return Plus(self, o)

    def __sub__(self, o):
        assert o is not None
        assert isinstance(o, Node)
        from . import Minus
        return Minus(self, o)

    def __mul__(self, o):
        assert o is not None
        assert isinstance(o, Node)
        from . import Times
        return Times(self, o)

    def __truediv__(self, o):
        assert o is not None
        assert isinstance(o, Node)
        from . import Divide
        return Divide(self, o)

    def __matmul__(self, o):
        assert o is not None
        assert isinstance(o, Node)
        from . import MatMul
        return MatMul(self, o)

    def __pow__(self, power, modulo=None):
        assert modulo is None
        assert power is not None
        assert isinstance(power, Node)
        from . import Pow
        return Pow(self, power)

    def __getitem__(self, index):
        assert index is not None
        assert (isinstance(index, int) or
                isinstance(index, slice) or
                isinstance(index, tuple))
        from . import Slice
        return Slice(self, index)

    def __init__(self, code, prior):
        assert code is not None
        assert isinstance(code, str)
        assert isinstance(prior, int) and prior >= 0
        self.__code = code
        self.__prior = prior
        self._gradient = lambda: None

    def forward(self):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError



