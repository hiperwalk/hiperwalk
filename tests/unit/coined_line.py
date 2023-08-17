import numpy as np
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
import hiperwalk as hpw
import unittest
from test_constants import *

class TestCoinedLine(unittest.TestCase):

    def setUp(self):
        self.num_vert = 41
        self.line = hpw.Line(self.num_vert)
        self.qw = hpw.Coined(self.line)

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_persistent_shift_right_state_transfer(self):
        # initial state in leftmost vertex
        # final state in rightmost vertex
        self.qw.set_shift('persistent')
        self.qw.set_coin('I')
        self.qw.set_marked([])

        init_state = self.qw.state([1, (0, 1)])

        num_steps = self.num_vert - 1
        final_state = self.qw.simulate(num_steps, init_state, hpc=False)
        final_state = final_state[0]

        self.assertTrue(
            final_state[-1] == 1 and np.all(final_state[:-1] == 0)
        )

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_persistent_shift_left_state_transfer(self):
        # initial state in leftmost vertex
        # final state in rightmost vertex
        self.qw.set_shift('persistent')
        self.qw.set_coin('I')
        self.qw.set_marked([])

        init_state = self.qw.state(
            [1, (self.num_vert - 1, self.num_vert - 2)])

        num_steps = self.num_vert - 1
        final_state = self.qw.simulate(num_steps, init_state, hpc=False)
        final_state = final_state[0]

        self.assertTrue(
            final_state[0] == 1 and np.all(final_state[1:] == 0)
        )

    @unittest.skipIf(not TEST_HPC, 'Skipping hpc tests.')
    def test_hpc_default_evolution_operator(self):

        num_steps = self.num_vert // 2
        center = self.num_vert // 2
        entries = [[1, (center, center + 1)],
                   [-1j, (center, center - 1)]]
        init_state = self.qw.state(*entries)

        states = self.qw.simulate((num_steps, 1), init_state, hpc=False)
        hpc_states = self.qw.simulate((num_steps, 1), init_state, hpc=True)

        self.assertTrue(
            np.allclose(states, hpc_states, rtol=1e-15, atol=1e-15)
        )
