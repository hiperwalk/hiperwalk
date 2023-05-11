import sys
sys.path.append('..')
import hiperwalk as hpw
import networkx as nx

num_vert = 10
adj_matrix = nx.adjacency_matrix(nx.cycle_graph(num_vert))
g = hpw.Graph(adj_matrix)
print(adj_matrix.indices)
print(adj_matrix.indptr)
print(list(map(g.degree, range(num_vert))))

coin = ['H'] * num_vert
coin[-1] = coin[0] = '-I'
qw = hpw.CoinedWalk(g, coin=coin)
print(qw._marked)
print(qw._marked_coin)

qw.set_marked([1, 2, 3])
print()
print(qw._marked)
print(qw._marked_coin)

qw.set_marked({'-H' : 1, '-G' : [3, 2], '-I' : 7})
print()
print(qw._marked)
print(qw._marked_coin)

qw.set_marked()
print()
print(qw._marked)
print(qw._marked_coin)
