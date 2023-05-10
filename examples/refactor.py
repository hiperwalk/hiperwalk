import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx

num_vert = 5
adj_matrix = nx.adjacency_matrix(nx.cycle_graph(num_vert))
g = hpw.Graph(adj_matrix)

print(g)
