from .graph import *

class SDMultigraph(Graph):
    r"""
    Class for managing symmetric directed multigraph.

    Functions for managing a symmetric directed multigraph are provided.
    Only a reference to the underlying multigraph is stored.
    """

    def __init__(self, graph):
        # underlying muligraph
        self.graph = graph

        # store the label of the first arc with tail in each vertex
        adj_matrix = graph.adjacency_matrix()
        data = adj_matrix.data
        first_arc = data.copy()

        degree = 0
        arc = 0
        for i in range(len(first_arc)):
            degree = first_arc[i]
            first_arc[i] = arc
            arc += degree
                

        print('-----------------------------')
        print(data)
        print(first_arc)
        print('-----------------------------')

        adj_matrix.data = first_arc
        print(adj_matrix.todense())
            

        # change the pointers again
