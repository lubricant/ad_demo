
from autodiff.node.node import Node
from autodiff.node.binary import Binary
from autodiff.node.unitary import Unitary

import numpy as np


class Softmax(Unitary):

    def __init__(self, log_prob):
        assert len(log_prob.shape) == 1
        super().__init__(log_prob, 'softmax')
        # log_prob should be 1-D array

    def __getitem__(self, correct):
        assert isinstance(correct, Node)
        return SoftmaxLoss(self, correct)

    def eval_op(self, log_prob):
        log_prob -= np.max(log_prob)
        norm_prob = np.exp(log_prob)
        return norm_prob / np.sum(norm_prob)

    def eval_grad(self, log_prob):
        log_prob -= np.max(log_prob)
        norm_prob = np.exp(log_prob)
        return norm_prob / np.sum(norm_prob)

    def backward(self, grad, from_softmax_loss=False):

        if not self._prepare_backward(grad):
            return

        if from_softmax_loss:
            pass


class SoftmaxLoss(Binary):

    def __init__(self, prob, correct):
        # correct should be int or 1-D array
        assert isinstance(prob, Softmax)
        assert len(prob.shape) == 1 and len(correct.shape) <= 1
        super().__init__(prob, correct, code='softmax-loss', prior=0)

        self._shape = ()  # loss result is a prob value
        self._correct_is_idx = not len(correct.shape)
        if self._correct_is_idx:
            assert np.size(prob) == np.size(correct)

    def eval_op(self, prob, correct):

        # actually, correct is not invoked in forward calculation
        if self._correct_is_idx:
            # correct should be a index of vector
            # which indicate the expected correct class
            assert isinstance(correct, int)
            assert 0 <= correct < np.size(prob)
            return -np.log(prob[correct])
        else:
            # correct should be binary(0/1) prob vector
            # which index of 1 indicated the expected correct class
            assert np.min(correct), np.max(correct) == (0, 1)
            assert np.sum(correct) == 1
            return -np.log(prob[np.argmax(correct)])

    def eval_grad(self, prob, correct):
        if self._correct_is_idx:
            # correct is a index of vector
            # which indicate the expected correct class
            assert isinstance(correct, int)
            assert 0 <= correct < np.size(prob)

            # equals to:
            # prob -= [..., 1, ...]
            prob[correct] -= 1.

        else:
            # correct is binary(0/1) prob vector like [..., 1, ...]
            # which index of 1 indicated the expected correct class
            assert np.min(correct), np.max(correct) == (0, 1)
            assert np.sum(correct) == 1

            prob -= correct

        # the error of the prob is the gradient of softmax
        return prob, 0.0  # gradient of correct is ignore
        # because it is not invoked in forward

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        assert isinstance(grad, (int, float))

        # keep _left_grad and _right_grad be zeros

        l_result, r_result = self._left.result, self._right.result
        l_grad, _ = self.eval_grad(np.copy(l_result), r_result)

        self._left.backward(l_grad, True)







