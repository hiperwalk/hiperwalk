import numpy as np
import networkx as nx
import random
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from test_constants import *
import hiperwalk as hpw
import unittest

class TestContinuousGraph(unittest.TestCase):

    def setUp(self):
        self.num_vert = 40
        nx_graph = nx.ladder_graph(self.num_vert // 2)
        self.adj = nx.adjacency_matrix(nx_graph)
        self.g = hpw.Graph(self.adj)
        self.gamma = 1/2
        self.qw = hpw.ContinuousTime(self.g, gamma=self.gamma)

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_hamiltonian_default(self):
        H = self.qw.get_hamiltonian()
        self.assertTrue((H - (-self.gamma*self.adj) != 0).nnz == 0)
        self.assertTrue(self.qw._hamiltonian is not None)
        self.assertTrue(self.qw._marked.size == 0)

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_hamiltonian_multiple_marked(self):
        prev_H = self.qw._hamiltonian
        prev_marked = self.qw._marked

        gamma = 1/3
        num_marked = random.randint(0, self.num_vert - 1)
        marked = random.sample(list(range(self.num_vert)), num_marked)
        self.qw.set_hamiltonian(gamma=gamma, marked=marked)

        H = self.qw.get_hamiltonian()

        marked_matrix = np.diag([1 if i in marked else 0
                                 for i in range(self.num_vert)])
        self.assertTrue(np.all(H - (-gamma*self.adj - marked_matrix) == 0))
        self.assertTrue(self.qw._hamiltonian is not None)
        self.assertTrue(id(prev_H) != id(self.qw._hamiltonian))
        self.assertTrue(self.qw._marked is not None)
        self.assertTrue(id(self.qw._marked) != id(prev_marked))
        
    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_evolution_operator_invalid_time(self):
        marked = self.qw._marked
        H = self.qw._hamiltonian
        U = self.qw._evolution

        self.assertRaises(ValueError, self.qw.set_evolution,
                          time=-1, gamma=self.qw.get_gamma(),
                          hpc=False)

        self.assertTrue(id(marked) == id(self.qw._marked))
        self.assertTrue(id(H) == id(self.qw._hamiltonian))
        self.assertTrue(id(U) == id(self.qw._evolution))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_evolution_operator_set_hamiltonian_no_marked(self):
        prev_marked = self.qw._marked
        prev_H = self.qw._hamiltonian
        prev_U = self.qw._evolution

        self.qw.set_evolution(time=1, hpc=False, gamma=1, marked=[])
        U = self.qw.get_evolution()

        self.assertTrue(U is not None)
        self.assertTrue(self.qw._evolution is not None)
        self.assertTrue(id(prev_marked) != id(self.qw._marked))
        self.assertTrue(id(prev_H) != id(self.qw._hamiltonian))
        self.assertTrue(id(prev_U) != id(self.qw._evolution))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_evolution_operator_set_hamiltonian_and_marked(self):
        prev_marked = self.qw._marked
        prev_H = self.qw._hamiltonian
        prev_U = self.qw._evolution

        self.qw.set_evolution(time=1, gamma=1, marked=[0], hpc=False)
        U = self.qw.get_evolution()

        self.assertTrue(U is not None)
        self.assertTrue(self.qw._evolution is not None)
        self.assertTrue(id(prev_marked) != id(self.qw._marked))
        self.assertTrue(id(prev_H) != id(self.qw._hamiltonian))
        self.assertTrue(id(prev_U) != id(self.qw._evolution))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_evolution_unitary(self):
        U = self.qw.set_evolution(gamma=self.qw.get_gamma(),
                time=1, hpc=False)
        U = self.qw.get_evolution()

        self.assertTrue(np.allclose(
            U@U.T.conjugate(), np.eye(U.shape[0])
        ))

    @unittest.skipIf(not TEST_HPC, 'Skipping hpc tests.')
    def test_hpc_evolution_unitary(self):
        self.qw.set_time(time=1, hpc=True)
        U = self.qw.get_evolution()

        self.assertTrue(np.allclose(
            U@U.T.conjugate(), np.eye(U.shape[0])
        ))
