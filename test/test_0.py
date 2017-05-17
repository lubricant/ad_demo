
import autodiff as ad

import numpy as np

a = ad.var('a', 1)
b = ad.var('b', 2)
c = a + b

ad.eval(c)
print(c)
print(c.result)
print(a.gradient, b.gradient)

print('--------------------------------')

c = ad.sin(c)
ad.eval(c)
print(c)
print(c.result)
print(a.gradient, b.gradient)

print('--------------------------------')

c = a + b
c = c * c
ad.eval(c)
print(c)
print(c.result)
print(a.gradient, b.gradient)

print('--------------------------------')
x = ad.var('x', np.array(np.arange(5)[::-1]))
y = ad.const(())
f = x[y]
print(f)
y.value = 1
ad.eval(f)
print(f.result)
print(f.gradient)


print('------------softmax-------------')
x = ad.var('x', np.array([[1, 2, 3], [4, 5, 3]], dtype=np.float64))
y = ad.const((2,3), 'labels')
p = ad.softmax(x, y)

y.value = np.array([[0, 1, 0], [0, 1, 0]])

print(p)
ad.eval(p)
print(p.result)
print(p.result.sum())
print(f.result)

print(f.gradient)
print(x.gradient)

print('------------1111111111-------------')
y = ad.const((2,), 'labels')
p = ad.softmax(x, y)
y.value = np.array([1, 1])

print(p)
ad.eval(p)
print(p.result)
print(p.result.sum())
print(f.result)

print(f.gradient)
print(x.gradient)


# print('--------------------------------')
# a = ad.var('a', np.zeros((1, 3, 7)))
# b = ad.var('b', np.zeros((6, 3, 3)))
# c = ad.conv(a, b, padding='SAME')
# print(c, c.shape)
# c = ad.conv(a, b, padding='VALID')
# print(c, c.shape)
#
# print('--------------------------------')
# a = ad.var('a', np.zeros((1, 3, 7, 7)))
# b = ad.var('b', np.zeros((6, 3, 3, 3)))
# c = ad.conv(a, b, padding=('SAME', 'SAME'))
# print(c, c.shape)
# c = ad.conv(a, b, padding='VALID')
# print(c, c.shape)

print('--------------------------------')

# a = np.array([[[1,1]]])[::,0]
# b = (np.arange(6) + 1).reshape((3,1,2))[..., ::-1]
# print(a)
# print(b)
# c = np.tensordot(a, b, ([-1], [-1]))
# print(c.shape)
# print(c)

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2],[3,4]])
c = a.reshape((2,1,1,3,1)) * b.reshape((2,1,1,1,2))
# c = np.multiply(a.reshape((2,3,1)), b.reshape((2,1,2)))
print(c.shape)
print(c)

print('--------------------------------')

a = ad.var('a', np.arange(3*4*5*6, dtype=np.float32).reshape((3,4,5,6)))

op = ad.prod

b = op(a, 0)
ad.eval(b)
# print(b.result)
print(b.gradient)


b = op(a, 1)
ad.eval(b)
# print(b.result)
print(b.gradient)


b = op(a, 2)
ad.eval(b)
# print(b.result)
print(b.gradient)

b = op(a, 3)
ad.eval(b)
# print(b.result)
print(b.gradient)
