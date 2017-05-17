import tensorflow as tf
import numpy as np

sess = tf.Session()


def test1d_conv(batch_, is_same, stride):
    in_ch_, out_ch_ = 1, 2
    sig_shape_ = (batch_, 5, in_ch_)
    flt_shape_ = (3, in_ch_, out_ch_)
    sig_in_ = (np.arange(np.prod(sig_shape_),dtype=np.float64) + 1).reshape(sig_shape_).tolist()
    flt_ke_ = (-(np.arange(np.prod(flt_shape_),dtype=np.float64) + 1).reshape(flt_shape_)).tolist()

    a = tf.Variable(sig_in_)
    b = tf.Variable(flt_ke_)
    c = tf.nn.conv1d(a, b, stride, 'SAME' if is_same else 'VALID')
    d = tf.gradients(c, [a, b])

    sess.run(tf.global_variables_initializer())
    e = sess.run(c)
    # print(e.shape)
    # print(e)
    f = sess.run(d)
    # print('input', f[0].shape)
    # print(f[0])
    print('kernel', f[1].shape)
    print(f[1])
    # print('---------------------------------------------')


# test1d_conv(2, False, 1)
# test1d_conv(2, True, 1)
# test1d_conv(2, False, 2)
# test1d_conv(2, True, 2)


def test2d_conv(batch_, is_same, stride):
    in_ch_, out_ch_ = 3, 6
    sig_shape_ = (batch_, 5, 5, in_ch_)
    flt_shape_ = (3, 3, in_ch_, out_ch_)
    sig_in_ = (np.arange(np.prod(sig_shape_),dtype=np.float64) + 1).reshape(sig_shape_).tolist()
    flt_ke_ = (-(np.arange(np.prod(flt_shape_),dtype=np.float64) + 1).reshape(flt_shape_)).tolist()
    strides = (1,stride,stride,1)

    a = tf.Variable(sig_in_)
    b = tf.Variable(flt_ke_)
    c = tf.nn.conv2d(a, b, strides, 'SAME' if is_same else 'VALID')
    d = tf.gradients(c, [a, b])

    sess.run(tf.global_variables_initializer())
    e = sess.run(c)
    # print(e.shape)
    # print(e)
    f = sess.run(d)
    # print('input', f[0].shape)
    # print(f[0])
    print('kernel', f[1].shape)
    print(f[1])
    # print('---------------------------------------------')

# test2d_conv(2, False, 1)
# test2d_conv(2, True, 1)
# test2d_conv(2, False, 2)
# test2d_conv(2, True, 2)


def test2d_pool(batch_, stride=None):
    k_size = 3
    sig_shape_ = (batch_, 5, 5, 2)
    sig_in_ = (np.arange(np.prod(sig_shape_),dtype=np.float64) + 1).reshape(sig_shape_)
    print(sig_in_)
    sig_in_ = sig_in_.tolist()

    a = tf.Variable(sig_in_)
    c = tf.nn.max_pool(a, ksize=[1, k_size, k_size, 1], strides=[1, k_size, k_size, 1], padding='VALID')
    d = tf.gradients(c, [a])

    sess.run(tf.global_variables_initializer())
    e = sess.run(c)
    print(e.shape)
    print(e)
    f = sess.run(d)
    print('grad', f[0].shape)
    print(f[0])
    # print('---------------------------------------------')

# test2d_pool(3)
# test2d_pool(2)


def test_reduce():
    a = tf.Variable([1., 2, 3, 2, 1.])
    b = tf.reduce_mean(a)
    c = tf.gradients(b, [a])

    sess.run(tf.global_variables_initializer())
    e = sess.run(c)

    print(e)
    print(c)

    b = tf.reduce_sum(a)
    c = tf.gradients(b, [a])

    sess.run(tf.global_variables_initializer())
    e = sess.run(c)

    print(e)
    print(c)

    b = tf.reduce_prod(a)
    c = tf.gradients(b, [a])

    sess.run(tf.global_variables_initializer())
    e = sess.run(c)

    print(e)
    print(c)

# test_reduce()


def test_reduce2(func):
    # a = tf.Variable([
    #     [1, 2, 3, 2, 1.],
    #     [2, 1, 5, 1, 2.]])
    #
    # b = func(a, reduction_indices=0)
    # c = func(a, reduction_indices=1)

    a = tf.Variable(np.arange(3*4*5*6, dtype=np.float32).reshape((3,4,5,6)))

    bb = func(a, reduction_indices=0)
    b = func(a, reduction_indices=1)
    c = func(a, reduction_indices=2)
    cc = func(a, reduction_indices=3)

    dd = tf.gradients(bb, [a])
    d = tf.gradients(b, [a])
    e = tf.gradients(c, [a])
    ee = tf.gradients(cc, [a])

    sess.run(tf.global_variables_initializer())
    ff = sess.run(dd)
    f = sess.run(d)
    g = sess.run(e)
    gg = sess.run(ee)

    # print(sess.run(bb))
    print(ff)

    # print(sess.run(b))
    print(f)

    # print(sess.run(c))
    print(g)

    # print(sess.run(cc))
    print(gg)

# test_reduce2(tf.reduce_mean)
# test_reduce2(tf.reduce_sum)

