import numpy as np
from sys import path as sys_path
from sys import argv
sys_path.append('../')
sys_path.append('../../')
import qwalk.coined as hpcoined
import unittest
from test_constants import *

class TestCoinedLine(unittest.TestCase):

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_initial_condition_in_middle(self):
        # middle right
        num_steps = 5
        entries = [[1, 0, 0]]
        line = hpcoined.Line(num_steps, entries)

        S = line.flip_flop_shift_operator()
        I = S@S
        init_state = line.simulate_walk(I)[0]

        self.assertTrue(
            np.all(init_state[:2*num_steps] == 0)
            and init_state[2*num_steps] == 1
            and np.all(init_state[2*num_steps + 1:] == 0)
        )


        # middle left
        num_steps = 5
        entries = [[1, 0, 1]]
        line = hpcoined.Line(num_steps, entries)

        init_state = line.simulate_walk(I)[0]

        self.assertTrue(
            np.all(init_state[:2*num_steps - 1] == 0)
            and init_state[2*num_steps - 1] == 1
            and np.all(init_state[2*num_steps:] == 0)
        )

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_persistent_shift_right_state_transfer(self):
        # initial state in leftmost vertex
        # final state in rightmost vertex

        num_steps = 5
        entries = [[1, 0, 0]]
        line = hpcoined.Line(num_steps, entries)

        S = line.persistent_shift_operator()
        final_state = line.simulate_walk(S)
        final_state = final_state[0]

        self.assertTrue(
            final_state[-1] == 1 and np.all(final_state[:-1] == 0)
        )

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_persistent_shift_left_state_transfer(self):
        # initial state in leftmost vertex
        # final state in rightmost vertex

        num_steps = 5
        entries = [[1, 0, 1]]
        line = hpcoined.Line(num_steps, entries)

        S = line.persistent_shift_operator()
        final_state = line.simulate_walk(S)
        final_state = final_state[0]

        self.assertTrue(
            final_state[0] == 1 and np.all(final_state[1:] == 0)
        )

    @unittest.skipIf(not TEST_HPC, 'Skipping hpc tests.')
    def test_hpc_default_evolution_operator(self):

        num_steps = 20
        entries = [[1, 0, 0], [-1j, 0, 1]]
        line = hpcoined.Line(num_steps, entries)
        U = line.evolution_operator()
        states = line.simulate_walk(U, save_interval=1, hpc=False)
        hpc_states = line.simulate_walk(U, save_interval=1, hpc=True)

        self.assertTrue(
            np.allclose(states, hpc_states, rtol=1e-15, atol=1e-15)
        )
