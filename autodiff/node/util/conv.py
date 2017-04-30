

from autodiff.node.node import Node
from autodiff.node.binary import Binary
from autodiff.node.unitary import Unitary

import numpy as np


class Convolute(Binary):

    def __init__(self, signals, filters, **args):
        '''
        计算卷积
            signals: 信号数据，格式为 [...(batch_size), in_channel, depth, height, width]
            filters: 滤波器（卷积核），格式为 [out_channel, in_channel, f_depth, f_height, f_width]
            args: 可选参数
                stride: 每次的运算，kernel 的移动距离
                padding: 填充

        需要注意的是，filters 的数据格式决定了卷积的计算方式（signals 的形状），对应关系如下：
            1-D 卷积：[out_channel, in_channel, f_width] => [..., in_channel, width]
            2-D 卷积：[out_channel, in_channel, f_height, f_width] => [..., in_channel, height, width]
            3-D 卷积：[out_channel, in_channel, f_depth, f_height, f_width] => [..., in_channel, depth, height, width]
            .....
        '''

        sig_shape, flt_shape = signals.shape, filters.shape
        sig_dim, flt_dim, kernel_dim = len(sig_shape), len(flt_shape), len(flt_shape) - 2
        assert 3 <= sig_dim == flt_dim <= 5  # 最多支持 3 维度

        kernel_shape = flt_shape[-kernel_dim:]
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


def slice_signal(sig_input, flt_kernel, stride, padding):

    kernel_shape, kernel_dim = flt_kernel.shape, len(flt_kernel.shape)
    assert 1 <= kernel_dim <= 3

    input_shape, input_dim = sig_input.shape, len(sig_input.shape)
    assert kernel_dim == input_dim == len(stride) == len(padding)

    k_width, k_height, k_depth = kernel_shape + (1,) * (3-kernel_dim)
    i_width, i_height, i_depth = input_shape + (1,) * (3-kernel_dim)
    p_width, p_height, p_depth = padding + (1,) * (3-kernel_dim)
    s_width, s_height, s_depth = stride + (1,) * (3-kernel_dim)

    assert not k_width % 2 and not k_height % 2 and not k_depth % 2
    kw_radius, kh_radius, kd_radius = k_width//2, k_height//2, k_depth//2

    o_width = (i_width + 2 * p_width - k_width) / s_width + 1
    o_height = (i_height + 2 * p_height - k_height) / s_height + 1
    o_depth = (i_depth + 2 * p_depth - k_depth) / s_depth + 1

    flat_kernel = flt_kernel.flatten()
    flat_input = np.zeros(k_width * k_height * k_depth)

    if kernel_dim == 1:
        output_buf = np.zeros(o_width)
        for w in range(o_width):
            iw_beg, iw_end = w - kw_radius, w + kw_radius
            if iw_beg < 0 or iw_end > i_width:
                flat_input[:] = 0

    if kernel_dim == 2:
        pass

    if kernel_dim == 3:
        pass


def guess_conv_op_result_shape(signal_shape, filter_shape, stride, padding):
    """
    signal_shape: (batch, in_channel) + input_shape
    filter_shape: (out_channel, in_channel) + kernel_shape

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

    input_shape = signal_shape[-(s_dim-2):]
    kernel_shape = filter_shape[-(f_dim-2):]

    # assert input_channel the same
    batch_size, sig_in_channel = signal_shape[:2]
    out_channel, flt_in_channel = filter_shape[:2]
    assert sig_in_channel == flt_in_channel

    output_shape = [batch_size, out_channel]
    for i in range(k_dim):
        input_size, kernel_size = input_shape[i], kernel_shape[i]
        assert input_size >= kernel_size

        stride_len, pad_size = stride[i], padding[i]
        assert input_size >= stride_len

        output_size = (input_size + 2 * pad_size - kernel_size) // stride_len + 1
        output_shape.append(output_size)

    return tuple(output_shape)

if __name__ == '__main__':
    a = np.arange(9).reshape((3, 3))
    b = a.flatten()
    print(a)
    b[2] = -1
    print(a)

