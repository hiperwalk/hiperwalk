from sys import path
path.append('..')
import hiperwalk as hpw

g = hpw.Hypercube(10)
dtqw = hpw.Coined(graph=g, marked={'-I': [0]})
l = {list(dtqw.ket(0)), list(dtqw.ket(1))}
print(type(l))
print(dtqw.success_probability(l))
print(type(l))
