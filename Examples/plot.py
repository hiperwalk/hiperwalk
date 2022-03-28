import networkx as nx
import sys
sys.path.append('..')
from CoinedModel import *
from PlotModule import *
from numpy import linspace

num_vert = 12

#G = nx.grid_graph(dim=(num_vert, num_vert), periodic=False)
#G = nx.complete_graph(num_vert)
G = nx.circular_ladder_graph(num_vert)

PlotDistributionProbabilityOnGraph(
    nx.adjacency_matrix(G), linspace(0, 1, num=G.number_of_nodes())
)
