
from autodiff.node.unitary import Unitary

from .conv import guess_conv_op_result_shape


class Pooling(Unitary):

    def __init__(self, conv, size, stride=None):
        '''
        池化操作
            conv: 卷积结果，形状为 [batch_size, depth, height, width, out_channel]
            size: 池化窗口大小，形状为 (d_size, h_size, w_size)
            stride: 池化步长，形状为 (d_stride, h_stride, w_stride)
                当 stride 为 None 时，令 stride 为 size
                使得每次窗口滑动取到的信息都不相互重叠
        '''

        conv_shape = conv.shape
        assert 2 < len(conv_shape) < 6

        out_ch = conv_shape[-1]
        out_shape = conv_shape[1:-1]
        assert len(out_shape) == len(size)

        if stride is None:
            stride = size
        padding = (0,) * len(stride)

        pool_shape = guess_conv_op_result_shape(conv_shape, size + (out_ch, out_ch), stride, padding)
        super().__init__(conv, 'pooling', pool_shape)

