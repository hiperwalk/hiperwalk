import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx

num_vert = 40000
adj_matrix = nx.adjacency_matrix(nx.cycle_graph(num_vert))
g = hpw.Graph(adj_matrix)

dtqw = hpw.CoinedWalk(g)
print(dtqw)
print(dtqw._shift.todense())
print(dtqw._coin.todense())
print(dtqw._marked)
