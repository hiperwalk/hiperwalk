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
        self.g = hpw.Hypercube(self.dim)

    def tearDown(self):
        del self.g

    def test_arc_direction(self):
        array = [self.g.arc_direction(a)
                 for a in range(self.g.number_of_arcs())]
        num_vert = self.g.number_of_vertices()
        self.assertTrue(array == list(range(self.dim))*num_vert)

    def test_arc_number(self):
        array = [self.g.arc_number((v, v ^ 1 << direction))
                 for v in range(self.g.number_of_vertices())
                 for direction in range(self.dim)]
        self.assertTrue(array == list(range(self.g.number_of_arcs())))

    def test_arc(self):
        array = [self.g.arc_number((self.g.arc(a)[0], self.g.arc(a)[1]))
                 for a in range(self.g.number_of_arcs())]
        self.assertTrue(array == list(range(self.g.number_of_arcs())))
