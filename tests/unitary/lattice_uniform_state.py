#import numpy as np
#import networkx as nx
#import sys
#sys.path.append('../..')
#import qwalk as hpw
#
## TODO: random graph
#grid_dim = 5
#grid = nx.grid_graph((grid_dim, grid_dim))
#coined_model = hpw.Coined(nx.adjacency_matrix(grid))
#uniform = coined_model.uniform_state()
#
#num_edges = len(grid.edges())
#print(
#    np.equal(
#        uniform,
#        np.ones(2*num_edges) / np.sqrt(2*num_edges)
#    ).all()
#)
