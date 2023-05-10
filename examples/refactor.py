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
print(qw._shift.todense())
print(qw._coin)
coin = qw.get_coin()
print()
print(coin.shape)
print(len(coin.shape))
qw.set_coin(coin)
print(qw._coin.todense())
