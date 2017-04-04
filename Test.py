
import autodiff as ad

s = set()
s.add((2,'b'))
s.add((1,'a'))
print(s)

s = sorted(s, key=lambda x:x[0])
print(s)
s.reverse()
print(s)