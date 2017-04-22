
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


print('--------------------------------')
x = ad.var('x', np.array([1, 2, 3]))
y = ad.const(())
p = ad.softmax(x)
f = p[y]
print(f)
y.value = 1
ad.eval(f)
print(p.result)
print(p.result.sum())
print(f.result)

print(f.gradient)
print(x.gradient)

print('------------1111111111-------------')
loss = - ad.log(p[y])
print(loss)
y.value = 1
ad.eval(loss)
print(p.gradient)

loss = - ad.log(p)[y]
print(loss)
y.value = 1
ad.eval(loss)
print(p.gradient)

print('--------------------------------')
x = ad.var('x',np.array(np.arange(36).reshape((4,3,3))))
print(x)
f = ad.maxout(x,2)
print(f)
ad.eval(f)
print(f.result)
print(f.gradient)