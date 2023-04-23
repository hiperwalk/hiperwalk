import numpy as np
import networkx as nx
import random
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from test_constants import *
import qwalk.continuous as ctqw
import unittest

num_vert = 20
test_graph = nx.ladder_graph(num_vert)
g = ctqw.Graph(nx.adjacency_matrix(test_graph))

class TestContinuousGraph(unittest.TestCase):

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_oracle_default(self):
        global g
        R = g.oracle()
        self.assertTrue(g._oracle == [0])
        R[0, 0] -= 1
        self.assertTrue(np.all(R == 0))
        
    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_oracle_no_marked_vertices(self):
        global g
        R = g.oracle(marked_vertices=None)
        self.assertTrue(g._oracle is None)
        self.assertTrue(R is None)

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_oracle_multiple_marked(self):
        global g
        global num_vert
        marked = random.sample(list(range(num_vert)),
                               random.randint(2, num_vert - 1))
        R = g.oracle(marked_vertices=marked)
        self.assertTrue(np.all(g._oracle == marked))
        for m in marked:
            R[m, m] -= 1
        self.assertTrue(np.all(R == 0))
