
from autodiff.node.node import Node
from autodiff.node.binary import Binary

import numpy as np


class SoftmaxLoss(Binary):

    def __init__(self, prob, correct):
        '''
        计算 Softmax
            prob: 原始概率，格式为 [batch_size, num_classes]
            correct: 期望输出
                当输入为类型的索引时，格式为 [batch_size]，取值范围 0 ~ num_classes-1 (int)
                当输入为期望概率时，格式为 [batch_size, num_classes]，取值范围为 0/1 (int) 且和为 1
        '''

        prob_shape, correct_shape = prob.shape, correct.shape

        assert len(prob_shape) == 2 and len(correct_shape) <= 2
        if len(prob_shape) == len(correct_shape):
            assert prob_shape == correct_shape
        else:
            assert prob_shape[0] == correct_shape[0]

        super().__init__(prob, correct, code='softmax_loss', prior=0,
                         guess_func=lambda a, b: prob_shape)

    @staticmethod
    def softmax(log_prob):

        log_prob = np.copy(log_prob)

        batch_size, _ = log_prob.shape
        line_wise = (batch_size, 1)
        sum_prob = max_prob = np.zeros(line_wise)  # share memory

        np.max(log_prob, axis=1, out=max_prob.ravel())
        log_prob -= max_prob

        norm_prob = np.exp(log_prob)
        np.sum(norm_prob, axis=1, out=sum_prob.ravel())
        norm_prob /= sum_prob

        return norm_prob

    @staticmethod
    def correct_idx(correct, prob_shape):

        batch_size, num_classes = prob_shape

        # each line of correct is binary(0/1) prob vector like [..., 1, ...]
        # which index of 1 indicated the expected correct class
        if len(correct.shape) == 2:
            max_idx = np.arange(batch_size)
            max_idx *= num_classes
            max_idx += np.argmax(correct, axis=1)
            correct = max_idx

        # each value correct is a index of vector between [0, num_classes-1]
        # which indicate the expected correct class
        assert len(correct.shape) == 1
        return correct

    def eval_op(self, prob, correct):

        prob = self.softmax(prob)
        correct = self.correct_idx(correct, prob.shape)

        return -np.log(prob.ravel()[correct])

    def eval_grad(self, prob, correct):

        prob = self.softmax(prob)
        correct = self.correct_idx(correct, prob.shape)
        prob.ravel()[correct] -= 1

        # the error of the prob is the gradient of softmax
        return prob, 0.0  # gradient of correct is ignore
        # because it is not invoked in forward

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        prob, correct = self._left.result, self._right.result
        softmax_grad, _ = self.eval_grad(prob, correct)

        softmax_grad *= grad
        self._left_grad += softmax_grad


if __name__ == "__main__":

    # log_prob_ = np.array([1, 2, 3])
    # log_prob_ -= np.max(log_prob_)
    # norm_prob_ = np.exp(log_prob_)
    # norm_prob_ /= np.sum(norm_prob_)
    #
    # correct_ = np.array([0, 1, 0])
    #
    # # one step
    # grad_one = np.copy(norm_prob_)
    # grad_one[np.argmax(correct_)] -= 1
    # print(grad_one)
    #
    # # two step
    # prob_mat_ = norm_prob_.reshape(norm_prob_.shape + (1,))
    # prob_eye_ = np.diag(norm_prob_)
    # grad_norm_ = prob_eye_ - prob_mat_ @ prob_mat_.T
    #
    # grad_loss_ = np.zeros(norm_prob_.shape)
    # grad_loss_[np.argmax(correct_)] = -1.0 / norm_prob_[np.argmax(correct_)]
    # grad_two = grad_norm_ @ grad_loss_
    # print(grad_two)
    #
    # print(np.outer(norm_prob_, norm_prob_))
    # print(prob_mat_ @ prob_mat_.T)

    # batch_size_, num_classes_ = 2, 3
    # line_wise_ = (batch_size_, 1)
    #
    # log_prob_ = np.array([[1, 2, 3],[4, 5, 3]])
    # max_prob_ = np.max(log_prob_, axis=1).reshape(line_wise_)
    # log_prob_ -= max_prob_
    #
    # norm_prob_ = np.exp(log_prob_)
    # sum_prob_ = np.sum(norm_prob_, axis=1).reshape(line_wise_)
    # norm_prob_ /= sum_prob_
    #
    # correct_ = np.array([[0, 1, 0],[0, 1, 0]])
    #
    # max_idx_ = np.arange(2)
    # max_idx_ *= 3
    # max_idx_ += np.argmax(correct_, axis=1)
    #
    # # one step
    # grad_one = np.copy(norm_prob_)
    # grad_one.ravel()[max_idx_] -= 1
    # print(grad_one)

    pass

