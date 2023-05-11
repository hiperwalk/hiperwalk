import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx
import numpy as np

num_vert = 5
adj_matrix = nx.adjacency_matrix(nx.cycle_graph(num_vert))
g = hpw.Graph(adj_matrix)

coin = ['H'] * num_vert
qw = hpw.CoinedWalk(g, coin=coin)
print(not np.any(
    qw.get_evolution().todense()
    - qw.get_shift() @ qw.get_final_coin()
))
