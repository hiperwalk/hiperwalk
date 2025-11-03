from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from test_constants import *
import unittest
import numpy as np
import hiperwalk as hpw

class TestCompleteBipartite(unittest.TestCase):
    
    def setUp(self):
        self.n1 = 10
        self.n2 = 20
        self.n = self.n1 + self.n2
        self.g = hpw.CompleteBipartite(self.n1, self.n2)

    def test_adjacency_matrix(self):
        A = np.array([[0]*self.n1 + [1]*self.n2]*self.n1
                     + [[1]*self.n1 + [0]*self.n2]*self.n2)
        A2 = self.g.adjacency_matrix()
        self.assertTrue(np.all(A - A2 == 0))

    def test_laplacian_matrix(self):
        D = np.diag([self.n2]*self.n1 + [self.n1]*self.n2)
        L = D - self.g.adjacency_matrix()
        L2 = self.g.laplacian_matrix()
        self.assertTrue(np.all(L - L2 == 0))

    def test_multigraph(self):
        adj = self.g.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.g._adj_matrix))

        adj[0, self.n1] = 2
        adj[self.n1, 0] = 2
        adj[0, self.n - 1] = 3
        adj[self.n - 1, 0] = 3

        mg = hpw.Multigraph(adj, copy=True)
        self.assertTrue(id(adj) != id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mg.adjacency_matrix()) == 0)

        mh = hpw.CompleteBipartite(self.n1, self.n2,
                                   multiedges=adj, copy=True)
        self.assertTrue(isinstance(mh, hpw.Multigraph))
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mh.adjacency_matrix()) == 0)

        d = {(0, self.n1): 2, (0, self.n - 1): 3}
        mh = hpw.CompleteBipartite(self.n1, self.n2, multiedges=d)
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mh.adjacency_matrix()) == 0)

        mg = hpw.Multigraph(adj, copy=False)
        self.assertTrue(id(adj) == id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))

        mh = hpw.CompleteBipartite(self.n1, self.n2,
                                   multiedges=adj, copy=False)
        self.assertTrue(id(adj) == id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))

        self.assertTrue(np.sum(mg._adj_matrix - mh._adj_matrix) == 0)

    def test_weighted_graph(self):
        adj = self.g.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.g._adj_matrix))

        adj = adj.astype(float)
        adj[0, self.n1] = 0.2
        adj[self.n1, 0] = 0.2
        adj[0, self.n - 1] = 0.3
        adj[self.n - 1, 0] = 0.3

        wg = hpw.WeightedGraph(adj, copy=True)
        self.assertTrue(id(adj) != id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(False)))
        self.assertTrue(np.sum(adj - wg._adj_matrix) == 0)

        wh = hpw.CompleteBipartite(self.n1, self.n2,
                                   weights=adj, copy=True)
        self.assertTrue(isinstance(wh, hpw.WeightedGraph))
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(False)))
        self.assertTrue(np.sum(adj - wh._adj_matrix) == 0)

        d = {(0, self.n1): 0.2, (0, self.n - 1): 0.3}
        wh = hpw.CompleteBipartite(self.n1, self.n2, weights=d)
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - wh._adj_matrix) == 0)

        wg = hpw.WeightedGraph(adj, copy=False)
        self.assertTrue(id(adj) == id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wg.adjacency_matrix(False)))

        wh = hpw.CompleteBipartite(self.n1, self.n2, weights=adj)
        self.assertTrue(id(adj) == id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wh.adjacency_matrix(False)))

        self.assertTrue(np.sum(wg._adj_matrix - wh._adj_matrix) == 0)
