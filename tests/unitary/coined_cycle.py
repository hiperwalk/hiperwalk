import numpy as np
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
import qwalk.coined as hpcoined
import unittest

class TestCoinedCycle(unittest.TestCase):

    def test_clockwise_roundabout(self):
        num_steps = 20
        cycle = hpcoined.Cycle(num_steps)

        S = cycle.persistent_shift_operator()
        init_state = cycle.state([(1, 0, 0)])

        final_state = cycle.simulate_walk(S, init_state, num_steps)[0]

        self.assertTrue(np.all(init_state == final_state))

    def test_anticlockwise_roundabout(self):
        num_steps = 20
        cycle = hpcoined.Cycle(num_steps)

        S = cycle.persistent_shift_operator()
        init_state = cycle.state([(1, 0, 1)])

        final_state = cycle.simulate_walk(S, init_state, num_steps)[0]

        self.assertTrue(np.all(init_state == final_state))
