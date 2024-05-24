from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from test_constants import *
import unittest
import numpy as np
import hiperwalk as hpw

class TestGrid(unittest.TestCase):
    
    def setUp(self):
        self.dim = 21

    def test_multigraph_natural_periodic(self):
        self.g = hpw.Grid(self.dim, periodic=True, diagonal=False)
        adj = self.g.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.g._adj_matrix))

        adj[0, 1] = 2
        adj[1, 0] = 2
        adj[1, 2] = 3
        adj[2, 1] = 3

        mg = hpw.Multigraph(adj, copy=True)
        self.assertTrue(id(adj) != id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mg.adjacency_matrix()) == 0)

        mh = hpw.Grid(self.dim, periodic=True, diagonal=False,
                      multiedges=adj, copy=True)
        self.assertTrue(isinstance(mh, hpw.Multigraph))
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mh.adjacency_matrix()) == 0)

        d = {(0, 1): 2, (1, 2): 3}
        mh = hpw.Grid(self.dim, periodic=True, diagonal=False,
                      multiedges=d)
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mh.adjacency_matrix()) == 0)

        mg = hpw.Multigraph(adj, copy=False)
        self.assertTrue(id(adj) == id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))

        mh = hpw.Grid(self.dim, periodic=True, diagonal=False,
                      multiedges=adj, copy=False)
        self.assertTrue(id(adj) == id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))

        self.assertTrue(np.sum(mg._adj_matrix - mh._adj_matrix) == 0)

    def test_weighted_graph_natural_periodic(self):
        self.g = hpw.Grid(self.dim, periodic=True, diagonal=False)
        adj = self.g.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.g._adj_matrix))

        adj = adj.astype(float)
        adj[0, 1] = 0.2
        adj[1, 0] = 0.2
        adj[1, 2] = 0.3
        adj[2, 1] = 0.3

        wg = hpw.WeightedGraph(adj, copy=True)
        self.assertTrue(id(adj) != id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(False)))
        self.assertTrue(np.sum(adj - wg._adj_matrix) == 0)

        wh = hpw.Grid(self.dim, periodic=True, diagonal=False,
                      weights=adj, copy=True)
        self.assertTrue(isinstance(wh, hpw.WeightedGraph))
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(False)))
        self.assertTrue(np.sum(adj - wh._adj_matrix) == 0)

        d = {(0, 1): 0.2, (1, 2): 0.3}
        wh = hpw.Grid(self.dim, periodic=True, diagonal=False,
                      weights=d)
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - wh._adj_matrix) == 0)

        wg = hpw.WeightedGraph(adj, copy=False)
        self.assertTrue(id(adj) == id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wg.adjacency_matrix(False)))

        wh = hpw.Grid(self.dim, periodic=True, diagonal=False,
                      weights=adj, copy=False)
        self.assertTrue(id(adj) == id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wh.adjacency_matrix(False)))

        self.assertTrue(np.sum(wg._adj_matrix - wh._adj_matrix) == 0)

    def test_multigraph_natural_not_periodic(self):
        self.g = hpw.Grid(self.dim, periodic=False, diagonal=False)
        self.g = hpw.Grid(self.dim, periodic=False, diagonal=False)
        adj = self.g.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.g._adj_matrix))

        adj[0, 1] = 2
        adj[1, 0] = 2
        adj[1, 2] = 3
        adj[2, 1] = 3

        mg = hpw.Multigraph(adj, copy=True)
        self.assertTrue(id(adj) != id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mg.adjacency_matrix()) == 0)

        mh = hpw.Grid(self.dim, periodic=False, diagonal=False,
                      multiedges=adj, copy=True)
        self.assertTrue(isinstance(mh, hpw.Multigraph))
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mh.adjacency_matrix()) == 0)

        d = {(0, 1): 2, (1, 2): 3}
        mh = hpw.Grid(self.dim, periodic=False, diagonal=False,
                      multiedges=d)
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mh.adjacency_matrix()) == 0)

        mg = hpw.Multigraph(adj, copy=False)
        self.assertTrue(id(adj) == id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))

        mh = hpw.Grid(self.dim, periodic=False, diagonal=False,
                      multiedges=adj, copy=False)
        self.assertTrue(id(adj) == id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))

        self.assertTrue(np.sum(mg._adj_matrix - mh._adj_matrix) == 0)

    def test_weighted_graph_natural_not_periodic(self):
        self.g = hpw.Grid(self.dim, periodic=False, diagonal=False)
        adj = self.g.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.g._adj_matrix))

        adj = adj.astype(float)
        adj[0, 1] = 0.2
        adj[1, 0] = 0.2
        adj[1, 2] = 0.3
        adj[2, 1] = 0.3

        wg = hpw.WeightedGraph(adj, copy=True)
        self.assertTrue(id(adj) != id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(False)))
        self.assertTrue(np.sum(adj - wg._adj_matrix) == 0)

        wh = hpw.Grid(self.dim, periodic=False, diagonal=False,
                      weights=adj, copy=True)
        self.assertTrue(isinstance(wh, hpw.WeightedGraph))
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(False)))
        self.assertTrue(np.sum(adj - wh._adj_matrix) == 0)

        d = {(0, 1): 0.2, (1, 2): 0.3}
        wh = hpw.Grid(self.dim, periodic=False, diagonal=False,
                      weights=d)
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - wh._adj_matrix) == 0)

        wg = hpw.WeightedGraph(adj, copy=False)
        self.assertTrue(id(adj) == id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wg.adjacency_matrix(False)))

        wh = hpw.Grid(self.dim, periodic=False, diagonal=False,
                      weights=adj, copy=False)
        self.assertTrue(id(adj) == id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wh.adjacency_matrix(False)))

        self.assertTrue(np.sum(wg._adj_matrix - wh._adj_matrix) == 0)

    def test_multigraph_diagonal_periodic(self):
        self.g = hpw.Grid(self.dim, periodic=True, diagonal=True)
        adj = self.g.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.g._adj_matrix))

        adj[0, self.dim + 1] = 2
        adj[self.dim + 1, 0] = 2
        adj[1, self.dim + 2] = 3
        adj[self.dim + 2, 1] = 3

        mg = hpw.Multigraph(adj, copy=True)
        self.assertTrue(id(adj) != id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mg.adjacency_matrix()) == 0)

        mh = hpw.Grid(self.dim, periodic=True, diagonal=True,
                      multiedges=adj, copy=True)
        self.assertTrue(isinstance(mh, hpw.Multigraph))
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mh.adjacency_matrix()) == 0)

        d = {(0, self.dim + 1): 2, (1, self.dim + 2): 3}
        mh = hpw.Grid(self.dim, periodic=True, diagonal=True,
                      multiedges=d)
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mh.adjacency_matrix()) == 0)

        mg = hpw.Multigraph(adj, copy=False)
        self.assertTrue(id(adj) == id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))

        mh = hpw.Grid(self.dim, periodic=True, diagonal=True,
                      multiedges=adj, copy=False)
        self.assertTrue(id(adj) == id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))

        self.assertTrue(np.sum(mg._adj_matrix - mh._adj_matrix) == 0)

    def test_weighted_graph_diagonal_periodic(self):
        self.g = hpw.Grid(self.dim, periodic=True, diagonal=True)
        adj = self.g.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.g._adj_matrix))

        adj = adj.astype(float)
        adj[0, self.dim + 1] = 0.2
        adj[self.dim + 1, 0] = 0.2
        adj[1, self.dim + 2] = 0.3
        adj[self.dim + 2, 1] = 0.3

        wg = hpw.WeightedGraph(adj, copy=True)
        self.assertTrue(id(adj) != id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(False)))
        self.assertTrue(np.sum(adj - wg._adj_matrix) == 0)

        wh = hpw.Grid(self.dim, periodic=True, diagonal=True,
                      weights=adj, copy=True)
        self.assertTrue(isinstance(wh, hpw.WeightedGraph))
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(False)))
        self.assertTrue(np.sum(adj - wh._adj_matrix) == 0)

        d = {(0, self.dim + 1): 0.2, (1, self.dim + 2): 0.3}
        wh = hpw.Grid(self.dim, periodic=True, diagonal=True,
                      weights=d)
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - wh._adj_matrix) == 0)

        wg = hpw.WeightedGraph(adj, copy=False)
        self.assertTrue(id(adj) == id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wg.adjacency_matrix(False)))

        wh = hpw.Grid(self.dim, periodic=True, diagonal=True,
                      weights=adj, copy=False)
        self.assertTrue(id(adj) == id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wh.adjacency_matrix(False)))

        self.assertTrue(np.sum(wg._adj_matrix - wh._adj_matrix) == 0)

    def test_multigraph_diagonal_not_periodic(self):
        self.g = hpw.Grid(self.dim, periodic=False, diagonal=True)
        self.g = hpw.Grid(self.dim, periodic=False, diagonal=True)
        adj = self.g.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.g._adj_matrix))

        adj[0, self.dim + 1] = 2
        adj[self.dim + 1, 0] = 2
        adj[1, self.dim + 2] = 3
        adj[self.dim + 2, 1] = 3

        mg = hpw.Multigraph(adj, copy=True)
        self.assertTrue(id(adj) != id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mg.adjacency_matrix()) == 0)

        mh = hpw.Grid(self.dim, periodic=False, diagonal=True,
                      multiedges=adj, copy=True)
        self.assertTrue(isinstance(mh, hpw.Multigraph))
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mh.adjacency_matrix()) == 0)

        d = {(0, self.dim + 1): 2, (1, self.dim + 2): 3}
        mh = hpw.Grid(self.dim, periodic=False, diagonal=True,
                      multiedges=d)
        self.assertTrue(id(adj) != id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - mh.adjacency_matrix()) == 0)

        mg = hpw.Multigraph(adj, copy=False)
        self.assertTrue(id(adj) == id(mg._adj_matrix))
        self.assertTrue(id(adj) != id(mg.adjacency_matrix()))

        mh = hpw.Grid(self.dim, periodic=False, diagonal=True,
                      multiedges=adj, copy=False)
        self.assertTrue(id(adj) == id(mh._adj_matrix))
        self.assertTrue(id(adj) != id(mh.adjacency_matrix()))

        self.assertTrue(np.sum(mg._adj_matrix - mh._adj_matrix) == 0)

    def test_weighted_graph_diagonal_not_periodic(self):
        self.g = hpw.Grid(self.dim, periodic=False, diagonal=True)
        adj = self.g.adjacency_matrix()
        self.assertTrue(id(adj) != id(self.g._adj_matrix))

        adj = adj.astype(float)
        adj[0, self.dim + 1] = 0.2
        adj[self.dim + 1, 0] = 0.2
        adj[1, self.dim + 2] = 0.3
        adj[self.dim + 2, 1] = 0.3

        wg = hpw.WeightedGraph(adj, copy=True)
        self.assertTrue(id(adj) != id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(False)))
        self.assertTrue(np.sum(adj - wg._adj_matrix) == 0)

        wh = hpw.Grid(self.dim, periodic=False, diagonal=True,
                      weights=adj, copy=True)
        self.assertTrue(isinstance(wh, hpw.WeightedGraph))
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(False)))
        self.assertTrue(np.sum(adj - wh._adj_matrix) == 0)

        d = {(0, self.dim + 1): 0.2, (1, self.dim + 2): 0.3}
        wh = hpw.Grid(self.dim, periodic=False, diagonal=True,
                      weights=d)
        self.assertTrue(id(adj) != id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix()))
        self.assertTrue(np.sum(adj - wh._adj_matrix) == 0)

        wg = hpw.WeightedGraph(adj, copy=False)
        self.assertTrue(id(adj) == id(wg._adj_matrix))
        self.assertTrue(id(adj) != id(wg.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wg.adjacency_matrix(False)))

        wh = hpw.Grid(self.dim, periodic=False, diagonal=True,
                      weights=adj, copy=False)
        self.assertTrue(id(adj) == id(wh._adj_matrix))
        self.assertTrue(id(adj) != id(wh.adjacency_matrix(True)))
        self.assertTrue(id(adj) == id(wh.adjacency_matrix(False)))

        self.assertTrue(np.sum(wg._adj_matrix - wh._adj_matrix) == 0)
