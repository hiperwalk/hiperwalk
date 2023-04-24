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
adj = nx.adjacency_matrix(test_graph)
g = ctqw.Graph(adj)

class TestContinuousGraph(unittest.TestCase):

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_oracle_default(self):
        global g
        R = g.oracle()
        self.assertTrue(g._oracle is not None)
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
        self.assertTrue(g._oracle is not None)
        for m in marked:
            R[m, m] -= 1
        self.assertTrue(np.all(R == 0))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_hamiltonian_default(self):
        global g
        prev_R = g._oracle
        prev_H = g._hamiltonian

        H = g.hamiltonian(1/2)

        global adj
        if prev_R is None: 
            self.assertTrue(np.all(H - (-1/2)*adj == 0))
        else:
            self.assertTrue(len((H - (-1/2*adj - prev_R)).data) == 0)
        self.assertTrue(g._hamiltonian is not None)
        self.assertTrue(id(prev_H) != id(g._hamiltonian))
        self.assertTrue(id(prev_R) == id(g._oracle))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_hamiltonian_no_marked(self):
        global g
        prev_H = g._hamiltonian

        H = g.hamiltonian(1/2, marked_vertices=None)

        global adj
        self.assertTrue(len((H - (-1/2)*adj).data) == 0)
        self.assertTrue(g._hamiltonian is not None)
        self.assertTrue(id(prev_H) != id(g._hamiltonian))
        self.assertTrue(g._oracle is None)

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_hamiltonian_multiple_marked(self):
        global g
        prev_H = g._hamiltonian
        prev_R = g._oracle

        H = g.hamiltonian(1/2, marked_vertices=[1, 2])

        global adj
        marked_matrix = np.diag([0] + [1]*2 + [0]*(adj.shape[0] - 3))
        self.assertTrue(np.all(H - (-0.5*adj - marked_matrix) == 0))
        self.assertTrue(g._hamiltonian is not None)
        self.assertTrue(id(prev_H) != id(g._hamiltonian))
        self.assertTrue(g._oracle is not None)
        self.assertTrue(id(g._oracle) != id(prev_R))

    
    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_laplacian_default(self):
        global g
        prev_R = g._oracle
        prev_H = g._hamiltonian


        H = g.hamiltonian(0.25, laplacian=True)

        global adj
        D = np.diag(adj.sum(axis=1))
        self.assertTrue(
            np.all(H - (-1/4)*(D - adj) == 0)
        )
        self.assertTrue(g._hamiltonian is not None)
        self.assertTrue(id(g._hamiltonian) != id(prev_H))
        self.assertTrue(id(g._oracle) == id(prev_R))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_laplacian_one_marked(self):
        global g
        prev_R = g._oracle
        prev_H = g._hamiltonian

        H = g.hamiltonian(1/2, marked_vertices=0)
        global adj
        marked_matrix = np.diag([1] + [0]*(adj.shape[0] - 1))
        self.assertTrue(np.all(H - (-adj/2 - marked_matrix == 0)))
        self.assertTrue(g._hamiltonian is not None)
        self.assertTrue(id(g._hamiltonian) != id(prev_H))
        self.assertTrue(g._oracle is not None)
        self.assertTrue(id(g._oracle) != id(prev_R))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_evolution_operator_no_hamiltonian(self):
        global adj
        g = ctqw.Graph(adj)
        
        R = g._oracle
        H = g._hamiltonian
        U = g._evolution_operator
        self.assertRaises(AttributeError, g.evolution_operator,
                          time=1, hpc=False)
        self.assertTrue(id(R) == id(g._oracle))
        self.assertTrue(id(H) == id(g._hamiltonian))
        self.assertTrue(id(U) == id(g._evolution_operator))
    
        
    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_evolution_operator_invalid_time(self):
        global g

        R = g._oracle
        H = g._hamiltonian
        U = g._evolution_operator

        self.assertRaises(ValueError, g.evolution_operator,
                          time=0, hpc=False)
        self.assertRaises(ValueError, g.evolution_operator,
                          time=-1, hpc=False)

        self.assertTrue(id(R) == id(g._oracle))
        self.assertTrue(id(H) == id(g._hamiltonian))
        self.assertTrue(id(U) == id(g._evolution_operator))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_evolution_operator_set_hamiltonian_no_oracle(self):
        global g

        prev_R = g._oracle
        prev_H = g._hamiltonian
        prev_U = g._evolution_operator

        U = g.evolution_operator(time=1, gamma=1, hpc=False)

        self.assertTrue(U is not None)
        self.assertTrue(g._evolution_operator is not None)
        self.assertTrue(id(prev_R) == id(g._oracle))
        self.assertTrue(id(prev_H) != id(g._hamiltonian))
        self.assertTrue(id(prev_U) != id(g._evolution_operator))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_evolution_operator_set_hamiltonian_and_oracle(self):
        global g

        prev_R = g._oracle
        prev_H = g._hamiltonian
        prev_U = g._evolution_operator

        U = g.evolution_operator(time=1, gamma=1, marked_vertices=0,
                                 hpc=False)

        self.assertTrue(U is not None)
        self.assertTrue(g._evolution_operator is not None)
        self.assertTrue(id(prev_R) != id(g._oracle))
        self.assertTrue(id(prev_H) != id(g._hamiltonian))
        self.assertTrue(id(prev_U) != id(g._evolution_operator))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_evolution_unitary(self):
        global g

        U = g.evolution_operator(time=1, gamma=1, marked_vertices=0,
                                 hpc=False)

        self.assertTrue(np.allclose(
            U@U.T.conjugate(), np.eye(U.shape[0])
        ))

