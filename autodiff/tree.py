
from .node import Node

import numpy as np


class Tree(object):

    _root = None
    _node_set = None

    def __init__(self, root):

        assert isinstance(root, Node)
        self._root = root

        node_set = set()

        def visit(node):
            node_set.add(node)
            if node.dependency:
                for dep in node.dependency:
                    visit(dep)

        visit(root)

        self._node_set = sorted(node_set, key=lambda node: node.depth)

    def exec(self, feed_dict=None):

        for node in self._node_set:
            node.forward()

        for node in self._node_set[::-1]:
            grad = node.gradient
            depend = node.dependency

            if not depend:
                continue

            if not grad:
                def init_grad(dep):
                    assert dep.shape is not None
                    return 1 if dep.shape == () else np.ones(dep.shape)
                grad = tuple(map(init_grad, depend))

            assert len(grad) == len(depend)

            for dep, g in zip(depend, grad):
                dep.backward(g)

 


