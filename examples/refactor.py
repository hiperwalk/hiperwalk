import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx
import numpy as np

dim = 4
l = hpw.Lattice(dim, periodic=False, diagonal=False)

print(l.arc_direction((5, 6)))
print(l.arc_direction((5, 4)))
print(l.arc_direction((5, 9)))
print(l.arc_direction((5, 1)))
print()
print(l.arc_direction((0, 1)))
print(l.arc_direction((0, 3)))
print(l.arc_direction((0, 4)))
print(l.arc_direction((0, 12)))

#qw = hpw.CoinedWalk(graph=l, coin='I')
#
#psi0 = qw.ket(0, dim + 1)
#states = qw.simulate((0, dim//2, 1), psi0, False)
#print(states)
#probs = qw.probability_distribution(states)
#
#hpw.plot_probability_distribution(probs, graph=l)
