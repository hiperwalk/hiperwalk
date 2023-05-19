import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx
import numpy as np

dim = 4
l = hpw.Lattice(dim, periodic=True, diagonal=True)
print(np.all(l.adj_matrix.todense() == l.adj_matrix.T.todense()))
print(l.adj_matrix)
print(l.adj_matrix.todense())
print(l.adj_matrix.indices)
print(l.adj_matrix.indptr)
