import numpy as np
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from test_constants import *
import hiperwalk as hpw
import unittest

class TestHypercube(unittest.TestCase):
    
    def setUp(self):
        self.dim = 10
        self.hypercube = hpw.Hypercube(self.dim)

    def tearDown(self):
        del self.hypercube

    def test_arc_direction(self):
        self.g = hpw.SDMultigraph(self.hypercube)
        array = [self.g._neighbor_index(*self.g.arc(a))
                 for a in range(self.g.number_of_arcs())]
        num_vert = self.g.number_of_vertices()
        self.assertTrue(array == list(range(self.dim))*num_vert)

    def test_arc_number(self):
        self.g = hpw.SDMultigraph(self.hypercube)
        array = [self.g.arc_number((v, v ^ 1 << direction))
                 for v in range(self.g.number_of_vertices())
                 for direction in range(self.dim)]
        self.assertTrue(array == list(range(self.g.number_of_arcs())))

    def test_arc(self):
        self.g = hpw.SDMultigraph(self.hypercube)
        array = [self.g.arc_number((self.g.arc(a)[0], self.g.arc(a)[1]))
                 for a in range(self.g.number_of_arcs())]
        self.assertTrue(array == list(range(self.g.number_of_arcs())))

    def test_multigraph(self):
        adj = self.hypercube.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.hypercube._adj_matrix))

        adj[0, 1] = 2
        adj[1, 0] = 2
        adj[0, 2] = 3
        adj[2, 0] = 3

        mg = hpw.Multigraph(adj, copy=True)
        self.assertTrue(id(adj) != id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))
        self.assertTrue((adj - mg.adjacency_matrix()).nnz == 0)

        mh = hpw.Hypercube(self.dim, multiedges=adj, copy=True)
        self.assertTrue(isinstance(mh, hpw.Multigraph))
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue((adj - mh.adjacency_matrix()).nnz == 0)

        d = {(0, 1): 2, (0, 2): 3}
        mh = hpw.Hypercube(self.dim, multiedges=d)
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue((adj - mh.adjacency_matrix()).nnz == 0)

        mg = hpw.Multigraph(adj, copy=False)
        self.assertTrue(id(adj) == id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))

        mh = hpw.Hypercube(self.dim, multiedges=adj, copy=False)
        self.assertTrue(id(adj) == id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))

        self.assertTrue((mg._adj_matrix - mh._adj_matrix).nnz == 0)

    def test_weighted_graph(self):
        adj = self.hypercube.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.hypercube._adj_matrix))

        adj = adj.astype(float)
        adj[0, 1] = 0.2
        adj[1, 0] = 0.2
        adj[0, 2] = 0.3
        adj[2, 0] = 0.3

        wg = hpw.WeightedGraph(adj, copy=True)
        self.assertTrue(id(adj) != id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(False)))
        self.assertTrue((adj - wg._adj_matrix).nnz == 0)

        wh = hpw.Hypercube(self.dim, weights=adj, copy=True)
        self.assertTrue(isinstance(wh, hpw.WeightedGraph))
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(False)))
        self.assertTrue((adj - wh._adj_matrix).nnz == 0)

        d = {(0, 1): 0.2, (0, 2): 0.3}
        wh = hpw.Hypercube(self.dim, weights=d)
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix()))
        self.assertTrue((adj - wh._adj_matrix).nnz == 0)

        wg = hpw.WeightedGraph(adj, copy=False)
        self.assertTrue(id(adj) == id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wg.adjacency_matrix(False)))

        wh = hpw.Hypercube(self.dim, weights=adj, copy=False)
        self.assertTrue(id(adj) == id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wh.adjacency_matrix(False)))

        self.assertTrue((wg._adj_matrix - wh._adj_matrix).nnz == 0)
