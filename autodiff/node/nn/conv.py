

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
    kw_radius, kh_radius, kd_radius = k_width//2, k_height//2, k_depth//2

    def slide_win_1d(clip_input_2d=None, clip_win_2d=None, iter_2d=()):
        w_iter = 0
        w_start, w_stop = 0 + kw_radius - p_width, i_width - kw_radius + p_width
        for w in range(w_start, w_stop, s_width):
            w_beg, w_end = w - kw_radius, w + kw_radius + 1
            if 0 <= w_beg and w_end <= i_width:
                input_idx = slice(w_beg, w_end)
                win_idx = slice(0, k_width)
            else:
                assert not (w_beg < 0 and w_end > i_width)
                if w_beg < 0:
                    input_idx = slice(0, w_end)
                    win_idx = slice(-w_beg, k_width)
                else:
                    input_idx = slice(w_beg, i_width)
                    win_idx = slice(0, i_width-w_end)

            if clip_input_2d is not None:
                if not isinstance(clip_input_2d, tuple):
                    input_idx = (clip_input_2d, input_idx)
                else:
                    input_idx = clip_input_2d + (input_idx,)

            if clip_win_2d is not None:
                if not isinstance(clip_input_2d, tuple):
                    win_idx = (clip_win_2d, win_idx)
                else:
                    win_idx = clip_win_2d + (win_idx,)

            yield input_idx, win_idx, iter_2d + (w_iter,)
            w_iter += 1

    def slide_win_2d(clip_input_3d=None, clip_win_3d=None, iter_3d=()):
        h_iter = 0
        h_start, h_stop = 0 + kh_radius - p_height, i_height - kh_radius + p_height
        for h in range(h_start, h_stop, s_height):
            h_beg, h_end = h - kh_radius, h + kh_radius + 1

            if 0 <= h_beg and h_end <= i_height:
                clip_input = slice(h_beg, h_end)
                clip_win = slice(0, k_height)
            else:
                assert not (h_beg < 0 and h_end > i_height)
                if h_beg < 0:
                    clip_input = slice(0, h_end)
                    clip_win = slice(-h_beg, k_height)
                else:
                    clip_input = slice(h_beg, i_height)
                    clip_win = slice(0, i_height-h_end)

            if clip_input_3d is not None:
                clip_input = (clip_input_3d, clip_input)

            if clip_win_3d is not None:
                clip_win = (clip_win_3d, clip_win)

            yield clip_input, clip_win, iter_3d + (h_iter,)
            h_iter += 1

    def slide_win_3d():
        d_iter = 0
        d_start, d_stop = 0 + kd_radius - p_depth, i_depth - kd_radius + p_depth
        for d in range(d_start, d_stop, s_depth):
            d_beg, d_end = d - kd_radius, d + kd_radius + 1

            if 0 <= d_beg and d_end <= i_height:
                clip_input = slice(d_beg, d_end)
                clip_win = slice(0, k_depth)
            else:
                assert not (d_beg < 0 and d_end > i_depth)
                if d_beg < 0:
                    clip_input = slice(0, d_end)
                    clip_win = slice(-d_beg, k_depth)
                else:
                    clip_input = slice(d_beg, i_depth)
                    clip_win = slice(0, i_depth-d_end)

            yield clip_input, clip_win, (d_iter,)
            d_iter += 1

    if kernel_dim == 1:
        return slide_win_1d()

    if kernel_dim == 2:
        for i_2d, w_2d, h in slide_win_2d():
            for idx in slide_win_1d(i_2d, w_2d, h):
                yield idx

    if kernel_dim == 3:
        for i_3d, w_3d, d in slide_win_3d():
            for i_2d, w_2d, h in slide_win_2d(i_3d, w_3d, d):
                for idx in slide_win_1d(i_2d, w_2d, h):
                    yield idx


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

        if win_size < sig_buf.size:
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

    grad_pad = []
    for i in range(len(padding)):
        in_pad = padding[i]
        if not in_pad:  # VALID
            grad_pad.append(kernel_shape[i] - 1)
        else:  # SAME
            grad_pad.append(in_pad)
    padding_g = tuple(grad_pad)

    grad_buf = np.zeros((batch_size,) + kernel_shape + (out_ch,))
    grad_axes = [ax+1 for ax in range(len(kernel_shape) + 1)]
    flt_axes = [ax for ax in range(len(kernel_shape) + 1)]

    sig_grad = np.zeros(sig_shape)
    batch_index = (slice(0, batch_size),)

    def fill_buf(i_idx, w_idx):
        if not isinstance(w_idx, tuple):
            win_size = w_idx.stop - w_idx.start
        else:
            win_size = 1
            for w in w_idx:
                win_size *= w.stop - w.start

        if win_size < grad_buf.size:
            grad_buf[:] = 0

        grad_buf[batch_index + w_idx] = gradients[batch_index + i_idx]

    def flush_buf(w_pos):
        print(grad_buf)
        conv_result = np.tensordot(grad_buf, filters_t, (grad_axes, flt_axes))
        assert conv_result.shape == (batch_size, in_ch)
        sig_grad[batch_index + w_pos] = conv_result

    for input_idx, win_idx, win_pos in slide_window(output_shape, kernel_shape, stride, padding_g):
        fill_buf(input_idx, win_idx)
        flush_buf(win_pos)

    return sig_grad


if __name__ == '__main__':

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
        # print(conv2d.shape)
        # print(conv2d)

        grad_shape_ = guess_conv_op_result_shape(sig_shape_, flt_shape_, strides_, padding_)
        out_grad_ = np.zeros(grad_shape_) + 1
        conv2d_g = calc_grad(out_grad_, sig_in_, flt_ke_, strides_, padding_)
        print(conv2d_g.shape)
        print(conv2d_g)

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

