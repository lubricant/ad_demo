
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

    def exec(self, eval_grad=True):

        for node in self._node_set:
            node.forward()

        if not eval_grad:
            return

        for node in self._node_set[::-1]:
            grad = node.gradient
            depend = node.dependency

            if not grad and depend:
                init_grad = np.ones(node.shape) if node.shape else 1
                node.backward(init_grad)
                grad = node.gradient

            if not depend:
                continue

            assert len(grad) == len(depend)

            for dep, g in zip(depend, grad):
                dep.backward(g)

 


