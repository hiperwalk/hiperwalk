import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx
import numpy as np

dim = 4
l = hpw.Lattice(dim, periodic=True, diagonal=True)

qw = hpw.CoinedWalk(graph=l, coin='I')

for arc in [(0, 5), (0, 13), (0, 7), (0, 15)]:
    psi0 = qw.ket(arc[0], arc[1])
    print(psi0)
    states = qw.simulate((0, 4, 1), psi0, False)
    probs = qw.probability_distribution(states)
    vertices = [np.where(prob == 1)[0][0] for prob in probs]
    print(vertices)
    print()

