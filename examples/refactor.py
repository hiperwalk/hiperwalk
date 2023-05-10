import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx

num_vert = 10
adj_matrix = nx.adjacency_matrix(nx.cycle_graph(num_vert))
g = hpw.Graph(adj_matrix)
print(g.number_of_vertices())
print(g.number_of_arcs())
print(g.number_of_edges())

qw = hpw.CoinedWalk(g)
print(qw._shift.todense())
