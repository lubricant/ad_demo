
import autodiff as ad

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