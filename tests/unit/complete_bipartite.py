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
