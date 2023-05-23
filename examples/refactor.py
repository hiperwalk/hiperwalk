import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx
import numpy as np

dim = 10
l = hpw.Lattice(dim, periodic=True, diagonal=True)

qw = hpw.CoinedWalk(graph=l, coin='I')

psi0 = qw.ket(0, dim + 1)
states = qw.simulate((0, dim//2, 1), psi0, False)
print(states)
probs = qw.probability_distribution(states)

hpw.plot_probability_distribution(probs, graph=l)
