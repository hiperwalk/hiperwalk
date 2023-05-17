import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx
import numpy as np

num_vert = 101
adj_matrix = nx.adjacency_matrix(nx.cycle_graph(num_vert))
#adj_matrix = scipy.sparse.csr_array(
#            [[0, 1, 1, 1, 1],
#             [1, 0, 1, 1, 0],
#             [1, 1, 0, 0, 0],
#             [1, 1, 0, 0, 0],
#             [1, 0, 0, 0, 0]]
#        )
g = hpw.Graph(adj_matrix)

qw = hpw.ContinuousWalk(graph=adj_matrix, gamma=0.35)
psi0 = qw.ket(num_vert//2)
print(psi0)
states = qw.simulate(time=num_vert//2, initial_condition=psi0)
probs = qw.probability_distribution(states)
hpw.plot_probability_distribution(probs)
