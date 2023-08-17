import numpy as np
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from test_constants import *
import hiperwalk as hpw
import unittest

class TestComplete(unittest.TestCase):
    
    def setUp(self):
        self.num_vert = 49
        self.g = hpw.Complete(self.num_vert)

    def tearDown(self):
        del self.g

    def test_arc_number(self):
        array = [self.g.arc_number((u, v))
                 for u in range(self.g.number_of_vertices())
                 for v in self.g.neighbors(u)]
        self.assertTrue(np.all(
            array == list(range(self.g.number_of_arcs()))))

    def test_arc(self):
        array = [self.g.arc_number((self.g.arc(a)[0], self.g.arc(a)[1]))
                 for a in range(self.g.number_of_arcs())]
        self.assertTrue(array == list(range(self.g.number_of_arcs())))

    def test_invertibility_of_arc_and_arc_number(self):
        arc_number1 = np.arange(self.g.number_of_arcs())
        arc_notation1 = list(map(self.g.arc, arc_number1))
        arc_number2 = list(map(self.g.arc_number, arc_notation1))
        arc_notation2 = list(map(self.g.arc, arc_number1))

        self.assertTrue(np.all(arc_number1 == arc_number2))
        self.assertTrue(np.all(arc_notation1 == arc_notation2))
