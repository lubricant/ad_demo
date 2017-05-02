

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

    def calc_conv(self):
        signals, filters = self._left, self._right
        sig_shape, flt_shape = signals.shape, filters.shape
        k_shape = flt_shape[-(len(flt_shape) - 2):]

        batch_size, in_channel = sig_shape[:2]
        out_channel = flt_shape[0]

        k_size = 0  # 一个 kernel 的大小
        for k in k_shape:
            k_size *= k

        # 输入图像的
        in_size = k_size * in_channel


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
    kernel_shape = filter_shape[:-(f_dim-2)]
    assert input_shape == kernel_shape

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


def im2col(sig_input, kernel_shape, stride, padding):
    '''
    将样本数据分解为多个 kernel_shape 对应的数据窗口，方便卷积计算
        sig_input: 单个样本数据，格式为：[i_depth, i_height, i_width, in_channel]
        kernel_shape: 卷积核形状，格式为：(k_depth, k_height, k_width)
        stride: 卷积滑动步长，格式为：(s_depth, s_height, s_width)
        padding: 输入边界填充量，格式为：(p_depth, p_height, p_width)
    '''

    sig_shape = sig_input.shape
    assert len(sig_shape) >= 2
    assert len(sig_shape) == len(kernel_shape) + 1

    input_shape, channel = sig_shape[:-1], sig_shape[-1]

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

    w_start, w_stop = 0 + kw_radius - p_width, i_width - kw_radius + p_width
    h_start, h_stop = 0 + kh_radius - p_height, i_height - kh_radius + p_height
    d_start, d_stop = 0 + kd_radius - p_depth, i_depth - kd_radius + p_depth

    # o_width = (i_width + 2 * p_width - k_width) // s_width + 1
    # o_height = (i_height + 2 * p_height - k_height) // s_height + 1
    # o_depth = (i_depth + 2 * p_depth - k_depth) // s_depth + 1
    #
    # print(o_width, o_height, o_depth)

    def img2col_1d(w_sig_input, w_sig_buf):

        w_sig_shape = w_sig_input.shape
        higher_dim = len(w_sig_shape) - 2

        for w in range(w_start, w_stop, s_width):
            # w_size = k_width * channel
            w_beg, w_end = w - kw_radius, w + kw_radius + 1
            if 0 <= w_beg and w_end <= i_width:
                if not higher_dim:
                    w_sig_buf[:] = w_sig_input[w_beg: w_end].ravel()
                elif higher_dim == 1:
                    w_sig_buf[:] = w_sig_input[:, w_beg: w_end].ravel()
                elif higher_dim == 2:
                    print(w_beg, w_end, w_sig_buf.shape, w_sig_input.shape)
                    w_sig_buf[:] = w_sig_input[:, :, w_beg: w_end].ravel()
            else:
                assert not (w_beg < 0 and w_end > i_width)

                w_sig_buf[:] = 0
                line_size = k_width * channel
                if w_beg < 0:
                    pad_size = -w_beg * channel
                    if not higher_dim:
                        w_sig_buf[pad_size:] = w_sig_input[:w_end].ravel()

                    elif higher_dim == 1:
                        for i in range(w_sig_shape[0]):
                            offset = i * line_size
                            w_sig_buf[offset + pad_size: offset + line_size] = w_sig_input[i, :w_end].ravel()

                    elif higher_dim == 2:
                        for j in range(w_sig_shape[0]):
                            for i in range(w_sig_shape[1]):
                                offset = i * line_size
                                w_sig_buf[offset + pad_size: offset + line_size] = w_sig_input[j, i, :w_end].ravel()
                else:
                    pad_size = (w_end - i_width) * channel
                    if not higher_dim:
                        w_sig_buf[:-pad_size] = w_sig_input[w_beg:].ravel()
                    elif higher_dim == 1:
                        for i in range(w_sig_input.shape[0]):
                            offset = i * line_size
                            w_sig_buf[offset: offset + line_size - pad_size] = w_sig_input[i, w_beg:].ravel()
                    elif higher_dim == 2:
                        for j in range(w_sig_input.shape[0]):
                            for i in range(w_sig_input[j].shape[0]):
                                offset = i * line_size
                                w_sig_buf[offset: offset + line_size - pad_size] = w_sig_input[j, i, w_beg:].ravel()

            yield w_sig_buf

    def img2col_2d(h_sig_input, h_sig_buf):

        higher_dim = len(h_sig_input.shape) - 3

        for h in range(h_start, h_stop, s_height):
            h_beg, h_end = h - kh_radius, h + kh_radius + 1

            if 0 <= h_beg and h_end <= i_height:
                if not higher_dim:
                    clip_sig_input = h_sig_input[h_beg: h_end]
                else:
                    clip_sig_input = h_sig_input[:, h_beg: h_end]

                clip_sig_buf = h_sig_buf
            else:
                assert not (h_beg < 0 and h_end > i_height)
                if not higher_dim:
                    if h_beg < 0:
                        h_skip = -h_beg * k_width * channel
                        clip_sig_input = h_sig_input[:h_end]
                        h_sig_buf[:h_skip] = 0
                        clip_sig_buf = h_sig_buf[h_skip:]
                    else:
                        h_skip = (h_end - i_height) * k_width * channel
                        clip_sig_input = h_sig_input[h_beg:]
                        h_sig_buf[-h_skip:] = 0
                        clip_sig_buf = h_sig_buf[:-h_skip]
                else:
                    if h_beg < 0:
                        # for ...
                        h_skip = -h_beg * k_width * channel
                        clip_sig_input = h_sig_input[:, :h_end]
                        h_sig_buf[:h_skip] = 0
                        clip_sig_buf = h_sig_buf[h_skip:]
                    else:
                        h_skip = (h_end - i_height) * k_width * channel
                        clip_sig_input = h_sig_input[:, h_beg:]
                        h_sig_buf[-h_skip:] = 0
                        clip_sig_buf = h_sig_buf[:-h_skip]

            print('+++++++++++++++++++++++++++++++++++++')
            print(clip_sig_input.shape, clip_sig_buf.shape)
            # print(h_skip, clip_sig_input.shape, clip_sig_buf.shape)
            for _ in img2col_1d(clip_sig_input, clip_sig_buf):
                yield h_sig_buf

    def img2col_3d(d_sig_input, d_sig_buf):
        for d in range(d_start, d_stop, s_depth):
            d_beg, d_end = d - kd_radius, d + kd_radius + 1

            if 0 <= d_beg and d_end <= i_height:
                clip_sig_input = d_sig_input[d_beg: d_end]
                clip_sig_buf = d_sig_buf
            else:
                assert not (d_beg < 0 and d_end > i_depth)
                if d_beg < 0:
                    d_skip = -d_beg * k_height * k_width * channel
                    clip_sig_input = d_sig_input[:d_end]
                    d_sig_buf[:d_skip] = 0
                    clip_sig_buf = d_sig_buf[d_skip:]
                else:
                    d_skip = (d_end - i_depth) * k_height * k_width * channel
                    clip_sig_input = d_sig_input[d_beg:]
                    d_sig_buf[-d_skip:] = 0
                    clip_sig_buf = d_sig_buf[:-d_skip]

            print('---------------------------------------------')
            print(clip_sig_input.shape, clip_sig_buf.shape)
            # print(d_skip, clip_sig_input.shape, clip_sig_buf.shape)
            for _ in img2col_2d(clip_sig_input, clip_sig_buf):
                yield d_sig_buf

    kernel_size = k_width * k_height * k_depth
    sig_buf = np.zeros(kernel_size * channel)

    if kernel_dim == 1:
        return img2col_1d(sig_input, sig_buf)

    if kernel_dim == 2:
        return img2col_2d(sig_input, sig_buf)

    if kernel_dim == 3:
        return img2col_3d(sig_input, sig_buf)


if __name__ == '__main__':
    # is_same = True
    # in_ch = 1
    # sig_in = (np.arange(7*in_ch) + 1).reshape((7, in_ch))
    # ken_sh = (3,)
    # print(sig_in)
    # print(ken_sh)
    # for buf in im2col(sig_in, ken_sh, (1,), (1,) if is_same else (0,)):
    #     print(buf.reshape((3, in_ch)))

    # is_same = True
    # in_ch = 2
    # sig_in = (np.arange(25*in_ch) + 1).reshape((5, 5, in_ch))
    # ken_sh = (3,3)
    # print(sig_in)
    # print(ken_sh)
    # for buf in im2col(sig_in, ken_sh, (1,1), (1,1) if is_same else (0,0)):
    #     print('---------------------')
    #     print(buf.reshape((3,3, in_ch)))

    is_same = True
    in_ch = 1
    sig_in = (np.arange(125*in_ch) + 1).reshape((5, 5, 5, in_ch))
    ken_sh = (3,3,3)
    print(sig_in)
    print(ken_sh)
    for buf in im2col(sig_in, ken_sh, (1,1,1), (1,1,1) if is_same else (0,0,0)):
        print('---------------------')
        print(buf.reshape((3,3,3, in_ch)))
