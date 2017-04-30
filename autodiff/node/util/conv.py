

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
                         guess_func=lambda a, b: guess_conv_op_result_shape(sig_shape, flt_shape, stride, padding))


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
    pass
    # N, F = 7, 3
    # S = 1
    # O = (N-F)/S + 1
    # print(O, (N-F)%S)
    # stride = 2
    # O = (N-F)/stride + 1
    # print(O, (N-F)%S)
    # stride = 3
    # O = (N-F)/stride + 1
    # print(O, (N-F)%S)
    #
    # # if same
    # P = (F-1)/2
    #
    # H, W = 7, 5
    # HF, WF = 1, 1
    #
    # print()
