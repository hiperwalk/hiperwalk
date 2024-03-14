import numpy as np
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from test_constants import *
import hiperwalk as hpw
import unittest
import networkx as nx
import scipy.sparse

class TestGraph(unittest.TestCase):
    
    def setUp(self):
        self.adj_matrix = [
            [0, 0, 0, 0, 0, 0, 8, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 5, 0, 1],
            [0, 0, 0, 0, 1, 0, 2, 9, 0, 9, 2, 7, 0, 1, 0, 5, 1, 2, 0, 4],
            [0, 0, 0, 4, 0, 0, 7, 0, 0, 0, 0, 0, 0, 6, 1, 0, 0, 0, 3, 0],
            [0, 0, 1, 0, 2, 0, 6, 3, 0, 9, 9, 3, 0, 5, 0, 0, 5, 7, 8, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 9, 0, 1, 0, 0],
            [8, 0, 2, 7, 6, 0, 0, 6, 0, 8, 0, 0, 0, 0, 7, 0, 0, 3, 0, 0],
            [3, 0, 9, 0, 3, 0, 6, 0, 1, 6, 3, 0, 0, 2, 9, 6, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 2, 0, 0, 0, 9, 7, 3, 0, 3],
            [0, 0, 9, 0, 9, 1, 8, 6, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3],
            [0, 0, 2, 0, 9, 0, 0, 3, 0, 0, 2, 0, 4, 6, 0, 8, 0, 9, 0, 9],
            [0, 0, 7, 0, 3, 0, 0, 0, 2, 0, 0, 0, 3, 0, 6, 0, 6, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 7, 6, 0, 9, 0, 0, 0, 1],
            [0, 7, 1, 6, 5, 0, 0, 2, 0, 0, 6, 0, 6, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 7, 9, 0, 5, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 9, 0, 6, 9, 0, 8, 0, 9, 0, 6, 9, 0, 0, 0, 8],
            [0, 0, 1, 0, 5, 0, 0, 0, 7, 0, 0, 6, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 5, 2, 0, 7, 1, 3, 5, 3, 0, 9, 0, 0, 0, 0, 0, 4, 0, 8, 4],
            [0, 0, 0, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
            [4, 1, 4, 0, 0, 0, 0, 0, 3, 3, 9, 5, 1, 0, 0, 8, 0, 4, 0, 0]]

        self.simple_adj = [[1 if self.adj_matrix[row][col] != 0 else 0
                               for col in range(len(self.adj_matrix))]
                           for row in range(len(self.adj_matrix))]

        self.g = hpw.Multigraph(self.adj_matrix)
        self.simple_g = hpw.Multigraph(self.simple_adj)

    def test_init_with_python_matrix(self):
        # multigraph
        g = hpw.Multigraph(self.adj_matrix)
        adj_matrix = g.adjacency_matrix()
        self.assertTrue(scipy.sparse.issparse(adj_matrix))

        adj_matrix = adj_matrix.todense()
        self.assertTrue(np.all(self.adj_matrix == adj_matrix))

        # simple graph
        g = hpw.Multigraph(self.simple_adj)
        simple_adj = g.adjacency_matrix()
        self.assertTrue(scipy.sparse.issparse(simple_adj))

        simple_adj = simple_adj.todense()
        self.assertTrue(np.all(self.simple_adj == simple_adj))

    def test_init_with_numpy_matrix(self):
        # multigraph
        numpy_matrix = np.array(self.adj_matrix)
        g = hpw.Multigraph(numpy_matrix)
        adj_matrix = g.adjacency_matrix()
        self.assertTrue(scipy.sparse.issparse(adj_matrix))

        adj_matrix = adj_matrix.todense()
        self.assertTrue(np.all(numpy_matrix == adj_matrix))

        # simple graph
        numpy_matrix = np.array(self.simple_adj)
        g = hpw.Multigraph(numpy_matrix)
        simple_adj = g.adjacency_matrix()
        self.assertTrue(scipy.sparse.issparse(simple_adj))

        simple_adj = simple_adj.todense()
        self.assertTrue(np.all(numpy_matrix == simple_adj))

    def test_init_with_csr_array(self):
        # multigraph
        sparse_matrix = scipy.sparse.csr_array(self.adj_matrix)
        g = hpw.Multigraph(sparse_matrix, copy=True)
        adj_matrix = g.adjacency_matrix()
        self.assertTrue(scipy.sparse.issparse(adj_matrix))

        self.assertTrue((sparse_matrix - adj_matrix).nnz == 0)

        # simple graph
        sparse_matrix = scipy.sparse.csr_array(self.simple_adj)
        g = hpw.Multigraph(sparse_matrix, copy=True)
        simple_adj = g.adjacency_matrix()
        self.assertTrue(scipy.sparse.issparse(simple_adj))

        self.assertTrue((sparse_matrix - simple_adj).nnz == 0)
    
    def test_init_with_networkx_graph(self):
        # multigraph
        numpy_matrix = np.array(self.adj_matrix)
        graph = nx.Graph(numpy_matrix)
        g = hpw.Multigraph(graph)
        adj_matrix = g.adjacency_matrix()
        self.assertTrue(scipy.sparse.issparse(adj_matrix))

        adj_matrix = adj_matrix.todense()
        self.assertTrue(np.all(self.adj_matrix == adj_matrix))

        # simples graph
        numpy_matrix = np.array(self.simple_adj)
        graph = nx.Graph(numpy_matrix)
        g = hpw.Multigraph(graph)
        simple_adj = g.adjacency_matrix()
        self.assertTrue(scipy.sparse.issparse(simple_adj))

        simple_adj = simple_adj.todense()
        self.assertTrue(np.all(self.simple_adj == simple_adj))

    def test_number_of_edges(self):
        num_vert = len(self.adj_matrix)

        # multigraph
        nx_g = nx.MultiGraph()
        nx_g.add_edges_from([
            (u, v)
            for u in range(num_vert)
            for v in range(u, num_vert)
            for w in range(self.adj_matrix[u][v])
        ])
        num_edges1 = self.g.number_of_edges()
        num_edges2 = nx_g.number_of_edges()
        self.assertTrue(num_edges1 == num_edges2)

        # simple graph
        nx_g = nx.Graph(np.array(self.simple_adj))
        num_edges1 = self.simple_g.number_of_edges()
        num_edges2 = nx_g.number_of_edges()
        self.assertTrue(num_edges1 == num_edges2)

        # multiedges simultaneously incident to two vertices
        for u in range(num_vert):
            for v in range(num_vert):
                ret = self.g.number_of_edges(u, v)
                self.assertTrue(ret == self.adj_matrix[u][v])
                ret = self.simple_g.number_of_edges(u, v)
                self.assertTrue(ret == self.simple_adj[u][v])

    def test_number_of_nodes(self):
        num_vert = len(self.adj_matrix)
        self.assertTrue(self.g.number_of_vertices() == num_vert)
        self.assertTrue(self.simple_g.number_of_vertices() == num_vert)

    def test_degree(self):
        degrees1 = np.sum(np.array(self.adj_matrix), axis=1)
        degrees2 = [self.g.degree(v)
                    for v in range(self.g.number_of_vertices())]
        self.assertTrue(np.all(degrees1 - degrees2 == 0))

        degrees1 = np.sum(np.array(self.simple_adj), axis=1)
        degrees2 = [self.simple_g.degree(v)
                    for v in range(self.simple_g.number_of_vertices())]
        self.assertTrue(np.all(degrees1 - degrees2 == 0))
