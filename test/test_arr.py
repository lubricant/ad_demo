
import autodiff as ad

a = ad.var('a', [1,2,3])
# b = ad.var('b', 2)
b = ad.var('b', [4,5,6])
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

c = a * b[1]
ad.eval(c)
print(c)
print(c.result)
print(a.gradient, b.gradient)

print('--------------------------------')

c = a @ b
ad.eval(c)
print(c)
print(c.result)
print(a.gradient, b.gradient)

print('--------------------------------')
x, one = ad.var('x', 3), ad.const(1)
y = one / (one + ad.exp(-x))
ad.eval(y)
print(y)
print(y.result)
print(x.gradient)
print((1-y.result)*y.result)
