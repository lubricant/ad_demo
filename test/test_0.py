
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
x = ad.var('x', np.array(np.arange(5)))
y = ad.const(())
f = ad.softmax(x, y)
print(f)
y.value = 1
ad.eval(f)
print(np.exp(-f.result).sum())
print(f.result)
print(f.gradient)

y.value = 3
ad.eval(f)
print(f.result)
print(f.gradient)


print('--------------------------------')
x = ad.var('x',np.array(np.arange(36).reshape((4,3,3))))
print(x)
f = ad.maxout(x,2)
print(f)
ad.eval(f)
print(f.result)
print(f.gradient)