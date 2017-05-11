
from autodiff.node.unitary import Unitary

# from .conv import slide_window

import numpy as np


class Pooling(Unitary):

    def __init__(self, name, conv, size, stride=None):
        '''
        池化操作
            conv: 卷积结果，形状为 [batch_size, depth, height, width, channel]
            size: 池化窗口大小，形状为 (d_size, h_size, w_size)
            stride: 池化步长，形状为 (d_stride, h_stride, w_stride)
                当 stride 为 None 时，令 stride 为 size
                使得每次窗口滑动取到的信息都不相互重叠
        '''

        assert size is not None

        conv_shape = conv.shape
        assert 2 < len(conv_shape) < 6

        out_shape = conv_shape[1:-1]
        assert len(out_shape) == len(size)

        if stride is None:
            stride = size

        self.size = size
        self.stride = stride
        super().__init__(conv, '%s-pool' % name,
                         guess_pool_op_result_shape(conv_shape, size, stride))


class MaxPool(Pooling):

    def __init__(self, conv, **args):
        self.max_indices = None
        super().__init__('max', conv, **args)

    def eval_op(self, operand):
        max_idx, max_pool = max_sampling(operand, self.size, self.stride)
        self.max_indices = max_idx
        return max_pool

    def backward(self, grad):

        if not self._prepare_backward(grad):
            return

        conv_shape, conv_grad = self._operand.shape, self._op_grad
        calc_max_sampling_grad(conv_shape, self.size, self.stride,
                               grad, self.max_indices, conv_grad)


def guess_pool_op_result_shape(conv_shape, kernel_shape, stride):
    """
    计算池化结果的形状，为了方便计算，对输入的形状进行分解：

    1-D: conv_shape = (batch_size, i_width, channel)
         kernel_shape = (k_width)
         stride: (stride_w)

    2-D: conv_shape = (batch_size, i_height, i_width, channel)
         kernel_shape = (k_height, k_width)
         stride: (stride_h, stride_w)

    3-D: conv_shape = (batch_size, i_depth, i_height, i_width, channel)
         kernel_shape = (k_depth, k_height, k_width)
         stride: (stride_d, stride_h, stride_w)
    """

    k_dim = len(kernel_shape)
    assert 1 <= k_dim <= 3
    assert k_dim == len(stride) == len(conv_shape) - 2

    input_shape = conv_shape[1:-1]
    batch_size, channel = conv_shape[0], conv_shape[-1]

    output_shape = [batch_size]
    for i in range(len(kernel_shape)):
        input_size, kernel_size = input_shape[i], kernel_shape[i]
        assert input_size >= kernel_size

        stride_len = stride[i]
        assert input_size >= stride_len

        output_size = (input_size - kernel_size) // stride_len + 1
        output_shape.append(output_size)

    output_shape.append(channel)

    return tuple(output_shape)


def slide_window(input_shape, kernel_shape, stride):
    '''
        滑动窗口，计算并返回池化计算所需的索引
        input_shape: 单个样本数据，格式为：(i_depth, i_height, i_width)
        kernel_shape: 池化核形状，格式为：(k_depth, k_height, k_width)
        stride: 卷积滑动步长，格式为：(s_depth, s_height, s_width)
        padding: 输入边界填充量，格式为：(p_depth, p_height, p_width)
    '''

    kernel_dim = len(kernel_shape)
    assert 1 <= kernel_dim <= 3

    input_dim = len(input_shape)
    assert kernel_dim == input_dim == len(stride)

    k_width, k_height, k_depth = kernel_shape + (1,) * (3-kernel_dim)
    i_width, i_height, i_depth = input_shape + (1,) * (3-kernel_dim)
    s_width, s_height, s_depth = stride + (1,) * (3-kernel_dim)

    def slide(
            input_size, win_size, step,
            clip_input=(), clip_win=(), prev_iter=()):

        n, start, stop = 0, 0, input_size - (input_size % step)
        for i in range(start, stop, step):
            beg, end = i, i + win_size

            if 0 <= beg and end <= input_size:
                input_idx = slice(beg, end)
                win_idx = slice(0, win_size)
            else:
                assert not (beg < 0 and end > input_size)
                if beg < 0:
                    input_idx = slice(0, end)
                    win_idx = slice(-beg, win_size)
                else:
                    input_idx = slice(beg, input_size)
                    win_idx = slice(0, input_size-end)

            yield clip_input + (input_idx,), clip_win + (win_idx,), prev_iter + (n,)
            n += 1

    if kernel_dim == 1:
        for win in slide(i_width, k_width, s_width):
            yield win

    if kernel_dim == 2:
        for win_2d in slide(i_height, k_height, s_height):
            for win in slide(i_width, k_width, s_width, *win_2d):
                yield win

    if kernel_dim == 3:
        for win_3d in slide(i_depth, k_depth, s_depth):
            for win_2d in slide(i_height, k_height, s_height, *win_3d):
                for win in slide(i_width, k_width, s_width, *win_2d):
                    yield win


def max_sampling(conv, size, stride):

    conv_shape = conv.shape
    input_shape = conv_shape[1:-1]
    assert len(size) == len(stride)

    batch_size, channel = conv_shape[0], conv_shape[-1]
    batch_idx = (slice(0, batch_size),)

    pool_shape = guess_pool_op_result_shape(conv_shape, size, stride)
    max_pool = np.zeros(pool_shape)

    idx_len = batch_size * channel
    max_idx = np.zeros(pool_shape + (idx_len,), np.int32)

    idx_buf = np.zeros((batch_size, channel), np.int32)
    flat_idx = idx_buf.ravel()

    conv_buf = np.zeros((batch_size,) + size + (channel,))
    conv_sample = conv_buf.reshape((batch_size, np.prod(size), channel)).swapaxes(-2, -1)
    flat_conv = conv_buf.ravel()

    base = np.arange(batch_size, dtype=np.int32)
    base *= np.prod(size) * channel
    base = np.outer(base, np.ones(channel, dtype=np.int32))
    base += np.arange(channel, dtype=np.int32)
    offset = base.ravel()

    for input_idx, _, win_pos in slide_window(input_shape, size, stride):
        np.copyto(conv_buf, conv[batch_idx + input_idx])
        np.argmax(conv_sample, axis=-1, out=idx_buf)
        flat_idx *= channel
        flat_idx += offset
        max_idx[batch_idx + win_pos] = flat_idx
        pool_win = max_pool[batch_idx + win_pos]
        pool_win[:] = flat_conv[flat_idx].reshape(pool_win.shape)

    del idx_buf, conv_buf, flat_conv

    return max_pool, max_idx


def calc_max_sampling_grad(conv_shape, size, stride, pool_grad, max_idx, conv_grad_out=None):

    input_shape = conv_shape[1:-1]
    assert len(size) == len(stride)

    batch_size, channel = conv_shape[0], conv_shape[-1]
    batch_idx = (slice(0, batch_size),)

    grad_buf = np.zeros((batch_size,) + size + (channel,))
    flat_buf = grad_buf.ravel()

    conv_grad = conv_grad_out if conv_grad_out is not None else np.zeros(conv_shape)
    assert conv_grad.shape == conv_shape

    for input_idx, _, win_pos in slide_window(input_shape, size, stride):
        grad_buf[:] = 0
        flat_idx = max_idx[batch_idx + win_pos]
        flat_buf[flat_idx] = pool_grad[batch_idx + win_pos].ravel()
        conv_grad[batch_idx + input_idx] += grad_buf

    del grad_buf, flat_buf

    return conv_grad


if __name__ == '__main__':

    def test2d(batch_):
        in_ch_ = 2
        sig_shape_ = (batch_, 5, 5, in_ch_)
        sig_in_ = (np.arange(np.prod(sig_shape_)) + 1).reshape(sig_shape_)
        print(sig_in_)
        # print(flt_ke_)
        # print(out_grad_)

        s = 3
        size_ = (s,) * 2
        strides_ = (s,) * 2
        max_pool_, max_idx_ = max_sampling(sig_in_, size_, strides_)
        print(max_pool_.shape)
        print(max_pool_)
        # print(max_idx_.shape)
        # print(max_idx_)
        print('--------------')
        pool_grad_ = np.ones(max_pool_.shape)
        conv_grad = calc_max_sampling_grad(sig_shape_, size_, strides_, pool_grad_, max_idx_)
        print(conv_grad.shape)
        print(conv_grad)

    # test2d(1)
    test2d(3)







