

from autodiff.node.node import Node
from autodiff.node.binary import Binary
from autodiff.node.unitary import Unitary

import numpy as np


class Convolute(Binary):

    def __init__(self, signals, filters, **args):
        '''
        计算卷积
            signals: 信号数据，格式为 [batch_size, depth, height, width, in_channel]
            filters: 滤波器（卷积核），格式为 [f_depth, f_height, f_width, in_channel, out_channel]
            args: 可选参数
                stride: 每次运算，kernel 的移动距离
                padding: 填充方式，可选项为 'SAME' 与 ’VALID‘，前者可以保证卷积前后结果的大小

        需要注意的是，filters 的数据格式决定了卷积的计算方式（signals 的形状），对应关系如下：
            1-D 卷积：conv1d ( [batch_size, width, in_channel], [f_width, in_channel, out_channel] )
            2-D 卷积：conv2d ( [batch_size, height, width, in_channel], [f_height, f_width, in_channel, out_channel] )
            3-D 卷积：conv3d ( [batch_size, depth, height, width, in_channel], [f_depth, f_height, f_width, in_channel, out_channel] )
        '''

        sig_shape, flt_shape = signals.shape, filters.shape
        sig_dim, flt_dim, kernel_dim = len(sig_shape), len(flt_shape), len(flt_shape) - 2
        assert 3 <= sig_dim == flt_dim <= 5  # 最多支持 3 维卷积

        kernel_shape = flt_shape[:-2]  # flt_shape - [in_channel, out_channel]
        assert np.sum(np.array(kernel_shape) % 2) == len(kernel_shape)
        # 卷积核的大小必须是奇数，方便后续处理

        # 计算滑动步长 stride 以及填充量 padding
        stride, padding = (1,) * kernel_dim, ('SAME',) * kernel_dim

        if 'stride' in args:
            stride_arg = args['stride']

            if isinstance(stride_arg, int):
                assert stride_arg > 0
                stride = (stride_arg,) * kernel_dim

            if isinstance(stride_arg, (list, tuple)):
                assert len(stride_arg) == kernel_dim
                for s in stride_arg:
                    assert s > 0
                stride = stride_arg

        if 'padding' in args:
            padding_arg = args['padding']

            if isinstance(padding_arg, str):
                assert padding_arg.upper() in ['VALID', 'SAME']
                padding = (padding_arg,) * kernel_dim

            if isinstance(padding_arg, (list, tuple)):
                assert len(padding_arg) == kernel_dim
                for p in padding_arg:
                    assert p.upper() in ['VALID', 'SAME']
                padding = padding_arg

        # 填充类型为 'SAME' 时会计算填充系数，使得输入与输出的大小: output_dim == input_dim
        # 填充类型为 'VALID' 时会使得填充系数为 0，输出的大小会缩小: output_dim < input_dim
        padding = [(k_size - 1) // 2 if pad.upper() == 'SAME' else 0
                   for pad, k_size in zip(padding, kernel_shape)]

        self.stride = stride
        self.padding = padding

        super().__init__(signals, filters, code='conv%dd' % kernel_dim, prior=0,
                         guess_func=lambda foo, bar: guess_conv_op_result_shape(sig_shape, flt_shape, stride, padding))

    def eval_op(self, left, right):
        pass

    def eval_grad(self, left, right):
        pass


def guess_conv_op_result_shape(signal_shape, filter_shape, stride, padding):
    """
    计算卷积结果的形状，为了方便计算，对输入的形状进行分解：
        signal_shape: (batch,) + input_shape + (in_channel,)
        filter_shape: kernel_shape + (in_channel, out_channel)

    input_shape 和 kernel_shape 是对齐的，方便后续处理
    可以分为下面不同的情况：

    1-D: input_shape = (i_width,)
         kernel_shape = (k_width,)
         stride: (stride_w,)
         padding: (padding_w,)

    2-D: input_shape = (i_height, i_width)
         kernel_shape = (k_height, k_width)
         stride: (stride_h, stride_w)
         padding: (padding_h, padding_w)

    3-D: input_shape = (i_depth, i_height, i_width)
         kernel_shape = (k_depth, k_height, k_width)
         stride: (stride_d, stride_h, stride_w)
         padding: (padding_d, padding_h, padding_w)
    """
    s_dim, f_dim, k_dim = len(signal_shape), len(filter_shape), len(filter_shape) - 2

    # assert input_channel the same
    batch_size, sig_in_channel = signal_shape[0], signal_shape[-1]
    out_channel, flt_in_channel = filter_shape[-1], filter_shape[-2]
    assert sig_in_channel == flt_in_channel

    input_shape = signal_shape[1:-1]
    kernel_shape = filter_shape[:-2]
    output_shape = [batch_size]
    for i in range(k_dim):
        input_size, kernel_size = input_shape[i], kernel_shape[i]
        assert input_size >= kernel_size

        stride_len, pad_size = stride[i], padding[i]
        assert input_size >= stride_len

        output_size = (input_size + 2 * pad_size - kernel_size) // stride_len + 1
        output_shape.append(output_size)

    output_shape.append(out_channel)

    return tuple(output_shape)


def slide_window(input_shape, kernel_shape, stride, padding):
    '''
        滑动窗口，计算并返回卷积计算所需的索引
        input_shape: 单个样本数据，格式为：(i_depth, i_height, i_width)
        kernel_shape: 卷积核形状，格式为：(k_depth, k_height, k_width)
        stride: 卷积滑动步长，格式为：(s_depth, s_height, s_width)
        padding: 输入边界填充量，格式为：(p_depth, p_height, p_width)
    '''

    kernel_dim = len(kernel_shape)
    assert 1 <= kernel_dim <= 3

    input_dim = len(input_shape)
    assert kernel_dim == input_dim == len(stride) == len(padding)

    k_width, k_height, k_depth = kernel_shape + (1,) * (3-kernel_dim)
    i_width, i_height, i_depth = input_shape + (1,) * (3-kernel_dim)
    p_width, p_height, p_depth = padding + (0,) * (3-kernel_dim)
    s_width, s_height, s_depth = stride + (1,) * (3-kernel_dim)
    assert k_width % 2 and k_height % 2 and k_depth % 2

    def slide(
            input_size, win_size, step, pad,
            clip_input=(), clip_win=(), prev_iter=()):

        radius = win_size // 2
        n, start, stop = 0, 0 + radius - pad, input_size - radius + pad
        for i in range(start, stop, step):
            beg, end = i - radius, i + radius + 1

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
        for win in slide(i_width, k_width, s_width, p_width):
            yield win

    if kernel_dim == 2:
        for win_2d in slide(i_height, k_height, s_height, p_height):
            for win in slide(i_width, k_width, s_width, p_width, *win_2d):
                yield win

    if kernel_dim == 3:
        for win_3d in slide(i_depth, k_depth, s_depth, p_depth):
            for win_2d in slide(i_height, k_height, s_height, p_height, *win_3d):
                for win in slide(i_width, k_width, s_width, p_width, *win_2d):
                    yield win


def calc_conv(signals, filters, stride, padding):
    '''
        计算卷积
        signals: 样本数据，格式为：[batch_size, i_depth, i_height, i_width, in_channel]
        filters: 滤波器，格式为：[k_depth, k_height, k_width, in_channel, out_channel]
        stride: 卷积滑动步长，格式为：(s_depth, s_height, s_width)
        padding: 输入边界填充量，格式为：(p_depth, p_height, p_width)
    '''

    sig_shape = signals.shape
    flt_shape = filters.shape
    out_shape = guess_conv_op_result_shape(sig_shape, flt_shape, stride, padding)

    input_shape = sig_shape[1:-1]
    kernel_shape = flt_shape[:-2]

    batch_size = sig_shape[0]
    in_ch, out_ch = flt_shape[-2], flt_shape[-1]

    sig_buf = np.zeros((batch_size,) + kernel_shape + (in_ch,))
    buf_size = np.prod((batch_size,) + kernel_shape)

    sig_axes = [ax+1 for ax in range(len(kernel_shape) + 1)]
    flt_axes = [ax for ax in range(len(kernel_shape) + 1)]

    conv_output = np.zeros(out_shape)
    batch_index = (slice(0, batch_size),)

    def fill_buf(i_idx, w_idx):
        if not isinstance(w_idx, tuple):
            win_size = w_idx.stop - w_idx.start
        else:
            win_size = 1
            for w in w_idx:
                win_size *= w.stop - w.start

        if win_size < buf_size:
            sig_buf[:] = 0

        sig_buf[batch_index + w_idx] = signals[batch_index + i_idx]

    def flush_buf(w_pos):
        conv_result = np.tensordot(sig_buf, filters, (sig_axes, flt_axes))
        assert conv_result.shape == (batch_size, out_ch)
        conv_output[batch_index + w_pos] = conv_result

    for input_idx, win_idx, win_pos in slide_window(input_shape, kernel_shape, stride, padding):
        fill_buf(input_idx, win_idx)
        flush_buf(win_pos)

    return conv_output


def calc_grad(gradients, signals, filters, stride, padding):
    '''
        累加梯度
        gradients: 上层梯度，格式为：[batch_size, o_depth, o_height, o_width, out_channel]
        signals: 样本数据，格式为：[batch_size, i_depth, i_height, i_width, in_channel]
        filters: 滤波器，格式为：[k_depth, k_height, k_width, in_channel, out_channel]
        stride: 卷积滑动步长，格式为：(s_depth, s_height, s_width)
        padding: 输入边界填充量，格式为：(p_depth, p_height, p_width)
    '''

    filters_t = filters.swapaxes(-2, -1)

    sig_shape = signals.shape
    flt_shape = filters.shape
    conv_shape = guess_conv_op_result_shape(sig_shape, flt_shape, stride, padding)
    assert gradients.shape == conv_shape

    output_shape = conv_shape[1:-1]
    input_shape = sig_shape[1:-1]
    kernel_shape = flt_shape[:-2]
    assert len(kernel_shape) == len(padding)

    batch_size = conv_shape[0]
    in_ch, out_ch = flt_shape[-2], flt_shape[-1]

    grad_buf = np.zeros((batch_size,) + kernel_shape + (out_ch,))
    buf_size = np.prod((batch_size,) + kernel_shape)

    grad_axes = [ax+1 for ax in range(len(kernel_shape) + 1)]
    flt_axes = [ax for ax in range(len(kernel_shape) + 1)]

    sig_grad = np.zeros(sig_shape)
    batch_index = (slice(0, batch_size),)

    def fill_buf(w_idx, w_pos):
        grad_buf[:] = gradients[batch_index + w_pos]

        if not isinstance(w_idx, tuple):
            win_size = w_idx.stop - w_idx.start
        else:
            win_size = 1
            for w in w_idx:
                win_size *= w.stop - w.start

        if win_size < buf_size:
            batch_index + w_idx
            grad_buf[:] = 0

    def flush_buf(i_idx):
        conv_result = np.tensordot(grad_buf, filters_t, (grad_axes, flt_axes))
        assert conv_result.shape == (batch_size, in_ch)
        sig_grad[batch_index + i_idx] += conv_result

    for input_idx, win_idx, win_pos in slide_window(input_shape, kernel_shape, stride, padding):
        fill_buf(win_idx)
        flush_buf(win_pos)

    return sig_grad


if __name__ == '__main__':

    def test1d(batch_, is_same_, stride):
        in_ch_, out_ch_ = 3, 6
        sig_shape_ = (batch_, 5, in_ch_)
        flt_shape_ = (3, in_ch_, out_ch_)

        sig_in_ = (np.arange(np.prod(sig_shape_)) + 1).reshape(sig_shape_)
        flt_ke_ = -(np.arange(np.prod(flt_shape_)) + 1).reshape(flt_shape_)
        # print(sig_in_)
        # print(flt_ke_)
        # print(out_grad_)

        padding_ = (1,) if is_same_ else (0,)
        strides_ = (stride,)
        conv2d = calc_conv(sig_in_, flt_ke_, strides_, padding_)
        print(conv2d.shape)
        print(conv2d)

    test1d(1, False, 1)

    def test2d(batch_, is_same_, stride):
        in_ch_, out_ch_ = 3, 6
        sig_shape_ = (batch_, 5, 5, in_ch_)
        flt_shape_ = (3, 3, in_ch_, out_ch_)

        sig_in_ = (np.arange(np.prod(sig_shape_)) + 1).reshape(sig_shape_)
        flt_ke_ = -(np.arange(np.prod(flt_shape_)) + 1).reshape(flt_shape_)
        # print(sig_in_)
        # print(flt_ke_)
        # print(out_grad_)

        padding_ = (1, 1) if is_same_ else (0, 0)
        strides_ = (stride, stride)
        conv2d = calc_conv(sig_in_, flt_ke_, strides_, padding_)
        print(conv2d.shape)
        print(conv2d)

        # grad_shape_ = guess_conv_op_result_shape(sig_shape_, flt_shape_, strides_, padding_)
        # out_grad_ = np.zeros(grad_shape_) + 1
        # conv2d_g = calc_grad(out_grad_, sig_in_, flt_ke_, strides_, padding_)
        # print(conv2d_g.shape)
        # print(conv2d_g)

    test2d(1, False, 1)
    # test2d(1, True, 1)
    # test2d(1, False, 2)
    # test2d(1, True, 2)

    # def test3d(batch_, is_same_, stride):
    #     in_ch_, out_ch_ = 3, 6
    #     sig_shape_ = (batch_, 5, 5, 5, in_ch_)
    #     flt_shape_ = (3, 3, 3, in_ch_, out_ch_)
    #     sig_in_ = (np.arange(np.prod(sig_shape_)) + 1).reshape(sig_shape_)
    #     flt_ke_ = (np.arange(np.prod(flt_shape_)) + 1).reshape(flt_shape_)
    #     padding_ = (1, 1, 1) if is_same_ else (0, 0, 0)
    #     strides = (stride, stride, stride)
    #     conv2d = calc_conv(sig_in_,flt_ke_, strides, padding_)
    #     print(conv2d.shape)
    #     print(conv2d)

    # test3d(1, False, 1)
    # test3d(1, True, 1)
    # test3d(1, False, 2)
    # test3d(1, True, 2)


