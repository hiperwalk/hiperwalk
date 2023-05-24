import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx
import numpy as np

dim = 4
l = hpw.Lattice(dim, periodic=False, diagonal=False)

print(l.adj_matrix.indptr)

print(l.arc_label(0, 1))
print(l.arc_label(0, 4))
print()
print(l.arc_label(1, 2))
print(l.arc_label(1, 0))
print(l.arc_label(1, 5))
print()
print(l.arc_label(2, 3))
print(l.arc_label(2, 1))
print(l.arc_label(2, 6))
print()
print(l.arc_label(3, 2))
print(l.arc_label(3, 7))
print()
print(l.arc_label(4, 5))
print(l.arc_label(4, 8))
print(l.arc_label(4, 0))
print()
print(l.arc_label(5, 6))
print(l.arc_label(5, 4))
print(l.arc_label(5, 9))
print(l.arc_label(5, 1))
print('======================')
print(l.arc_label(12, 13))
print(l.arc_label(12, 8))
print()
print(l.arc_label(13, 14))
print(l.arc_label(13, 12))
print(l.arc_label(13, 9))
print()
print(l.arc_label(14, 15))
print(l.arc_label(14, 13))
print(l.arc_label(14, 10))
print()
print(l.arc_label(15, 14))
print(l.arc_label(15, 11))

#qw = hpw.CoinedWalk(graph=l, coin='I')
#
#psi0 = qw.ket(0, dim + 1)
#states = qw.simulate((0, dim//2, 1), psi0, False)
#print(states)
#probs = qw.probability_distribution(states)
#
#hpw.plot_probability_distribution(probs, graph=l)
