
from autodiff.node.binary import Binary

import numpy as np


class Softmax(Binary):

    def __init__(self, a, b):
        super().__init__(a, b, code='softmax', prior=0)
        assert len(a.shape) == len(b.shape) == 1

    def eval_op(self, log_prob, correct):
        assert np.max(correct) == 1 and np.min(correct) == 0 and np.sum(correct) == 1
        log_prob -= np.max(log_prob)
        norm_prob = np.exp(log_prob)
        return -np.log(norm_prob / np.sum(norm_prob))

    def eval_grad(self, log_prob, correct):
        log_prob -= np.max(log_prob)
        norm_prob = np.exp(log_prob)
        norm_prob /= np.sum(norm_prob)
        return norm_prob - correct


