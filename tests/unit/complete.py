from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from test_constants import *
import unittest
import numpy as np
import hiperwalk as hpw

class TestCompleteBipartite(unittest.TestCase):
    
    def setUp(self):
        self.n = 50
        self.g = hpw.Complete(self.n)

    def test_adjacency_matrix(self):
        A = np.ones((self.n, self.n)) - np.eye(self.n)
        A2 = self.g.adjacency_matrix()
        self.assertTrue(np.all(A - A2 == 0))

    def test_laplacian_matrix(self):
        L = self.n*np.eye(self.n) - np.ones((self.n, self.n))
        L2 = self.g.laplacian_matrix()
        self.assertTrue(np.all(L - L2 == 0))
