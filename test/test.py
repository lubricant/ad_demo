
import autodiff as ad

a = ad.var('a', 1)
b = ad.var('b', 2)
c = a + b

ad.eval(c)
print(c.result)
print(c.gradient)
