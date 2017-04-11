
import autodiff as ad

a = ad.var('a', [[1,2,3],[4,5,6]])
b = ad.var('b', [1,2,3])

c = a + b
ad.eval(c)
print(c)
print(c.result)
print(a.gradient, b.gradient)

print('--------------------------------')

c = a * b
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
b = ad.var('b', [1,2])
c = b @ a
ad.eval(c)
print(c)
print(c.result)
print(a.gradient, b.gradient)


print('--------------------------------')
b = ad.var('b', [[5,5],[10,10],[15,15]])
c = a @ b
ad.eval(c)
print(c)
print(c.result)
print(a.gradient)
print(b.gradient)
print('--------------------------------')

c = b @ a
ad.eval(c)
print(c)
print(c.result)
print(a.gradient)
print(b.gradient)