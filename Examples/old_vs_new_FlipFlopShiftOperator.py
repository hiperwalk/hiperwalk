import numpy as np
import networkx as nx
import sys
sys.path.append('..')
from AuxiliaryFunctions import *
from CoinedModel import *

num_vert = 20
G = nx.grid_graph(dim=(num_vert, num_vert), periodic=True)
adj_matrix = nx.adjacency_matrix(G)

S = FlipFlopShiftOperator(adj_matrix)
#print(S)
S2 = NewFlipFlopShiftOperator(adj_matrix)
print( np.array_equal(S.data, S2.data) and
        np.array_equal(S.indices, S2.indices) and
        np.array_equal(S.indptr, S2.indptr) )

