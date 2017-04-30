

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
                stride: 移动距离每次的
                padding: 填充

        需要注意的是，filters 的数据格式决定了卷积的计算方式（signals 的形状），对应关系如下：
            1-D 卷积：[out_channel, in_channel, f_width] => [..., in_channel, width]
            2-D 卷积：[out_channel, in_channel, f_height, f_width] => [..., in_channel, height, width]
            3-D 卷积：[out_channel, in_channel, f_depth, f_height, f_width] => [..., in_channel, depth, height, width]
            .....
        '''

        sig_shape, flt_shape = signals.shape, filters.shape
        assert len(sig_shape) >= len(flt_shape) >= 3

        stride, padding = 1, 'same',

        if 'stride' in args:
            stride = args['stride']
            assert stride

        if 'padding' in args:
            padding = args['padding']
            assert padding in ['valid', 'same']

        kernel_dim = len(flt_shape) - 2  # ignore channel dim
        out_ch, in_ch = flt_shape[:kernel_dim]
        assert sig_shape[-kernel_dim] == flt_shape[-kernel_dim]

        # assert filter kernel has odd size
        assert np.sum(np.array(flt_shape) % 2) == len(flt_shape)

        if padding:
            pad = 0

        super().__init__(signals, filters, code='#', prior=0)


if __name__ == '__main__':

    def conv_shape(signal_shape, filter_shape, stride, padding):
        """
        signal_shape: (batch, in_channel) + input_shape
        filter_shape: (out_channel, in_channel) + kernel_shape
        padding: 'SAME' or 'VALID'

        1-D: input_shape = (i_width,)
             kernel_shape = (k_width,)
             stride: (stride_w,)
             padding: (?,)

        2-D: input_shape = (i_height, i_width)
             kernel_shape = (k_height, k_width)
             stride: (stride_h, stride_w)
             padding: (?, ?)

        3-D: input_shape = (i_depth, i_height, i_width)
             kernel_shape = (k_depth, k_height, k_width)
             stride: (stride_d, stride_h, stride_w)
             padding: (?, ?, ?)
        """
        s_dim, f_dim = len(signal_shape), len(filter_shape)
        assert 3 <= s_dim == f_dim <= 5

        input_shape = signal_shape[s_dim-2:]
        kernel_shape = filter_shape[f_dim-2:]

        # assert input_channel the same
        batch_size, sig_in_channel = signal_shape[:2]
        out_channel, flt_in_channel = filter_shape[:2]
        assert sig_in_channel == flt_in_channel

        k_dim = len(kernel_shape)
        assert k_dim == len(stride) == len(padding)
        assert np.sum(np.array(kernel_shape) % 2) == len(kernel_shape)
        # assert kernel has odd size

        output_shape = []
        for i in range(k_dim):
            input_size, kernel_size = input_shape[i], kernel_shape[i]
            assert input_size >= kernel_size

            stride_len, pad_size = stride[i], 0

            # if padding is 'SAME' then padding input_shape to keep: output_dim == input_dim
            # otherwise 'VALID' will cause: output_dim < input_dim
            if padding[i] == 'SAME':
                pad_size = (kernel_size - 1) // 2

            output_size = (input_size + 2 * pad_size - kernel_size) // stride_len + 1
            output_shape.append(output_size)

        return (batch_size, out_channel) + tuple(output_shape)


    in_shape = (1, 3, 7, 7)
    ke_shape = (6, 3, 3, 3)
    print(conv_shape(in_shape, ke_shape, padding=('SAME', 'SAME'), stride=(1,1)))
    print(conv_shape(in_shape, ke_shape, padding=('VALID', 'VALID'), stride=(1, 1)))
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
