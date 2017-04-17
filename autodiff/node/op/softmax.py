
from autodiff.node.binary import Binary

import numpy as np


class Softmax(Binary):

    correct_prob_is_idx = None

    def __init__(self, a, b):
        super().__init__(a, b, code='softmax', prior=0)
        assert len(a.shape) == 1 and len(b.shape) <= 1
        self.correct_prob_is_idx = not len(b.shape)
        if not self.correct_prob_is_idx:
            assert np.size(a) == np.size(b)

    def eval_op(self, log_prob, correct_prob):
        log_prob -= np.max(log_prob)
        norm_prob = np.exp(log_prob)
        return -np.log(norm_prob / np.sum(norm_prob))

    def eval_grad(self, log_prob, correct_prob):
        if self.correct_prob_is_idx:
            assert isinstance(correct_prob, int)
            assert 0 < correct_prob < np.size(log_prob)
        else:  # given a binary(0/1) prob vector
            assert np.min(correct_prob), np.max(correct_prob) == (0, 1)
            assert np.sum(correct_prob) == 1

        log_prob -= np.max(log_prob)
        norm_prob = np.exp(log_prob)
        norm_prob /= np.sum(norm_prob)
        return norm_prob - correct_prob, 0.

