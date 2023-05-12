import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx
import numpy as np
import scipy.sparse

num_vert = 5
adj_matrix = nx.adjacency_matrix(nx.cycle_graph(num_vert))
#adj_matrix = scipy.sparse.csr_array(
#            [[0, 1, 1, 1, 1],
#             [1, 0, 1, 1, 0],
#             [1, 1, 0, 0, 0],
#             [1, 1, 0, 0, 0],
#             [1, 0, 0, 0, 0]]
#        )
g = hpw.Graph(adj_matrix)

coin = ['G'] * num_vert
qw = hpw.CoinedWalk(g, coin=coin)
print(not np.any(
    qw.get_evolution().todense()
    - qw.get_shift() @ qw.get_final_coin()
))

num_arcs = g.number_of_arcs()
state = qw.uniform_state()
print(state)
print(qw.state([1, 0], [1, 1], [2, (1, 0)]))
print(qw.ket(0))
print(qw.ket(1, 0))
