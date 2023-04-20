import numpy as np
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
import qwalk.coined as hpcoined
import unittest
from test_constants import *

class TestCoinedSegment(unittest.TestCase):

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_persistent_shift_state_transfer(self):
        # initial state in leftmost vertex
        # final state in rightmost vertex

        num_vert = 10
        seg = hpcoined.Segment(num_vert)

        init_cond = seg.initial_condition([(1, 0, 0)])
        S = seg.persistent_shift_operator()
        seg.set_evolution_operator(S)
        seg.step(num_vert - 1)
        final_state = seg.simulate(hpc=False)
        final_state = final_state[0]

        self.assertTrue(
            final_state[-1] == 1 and np.all(final_state[:-1] == 0)
        )

    @unittest.skipIf(not TEST_HPC, 'Skipping hpc tests.')
    def test_hpc_persistent_shift_state_transfer(self):
        # initial state in leftmost vertex
        # final state in rightmost vertex

        num_vert = 10
        seg = hpcoined.Segment(num_vert)

        init_cond = seg.initial_condition([(1, 0, 0)])
        S = seg.persistent_shift_operator()
        seg.set_evolution_operator(S)
        seg.time(num_vert - 1)
        final_state = seg.simulate(hpc=True)
        final_state = final_state[0]

        self.assertTrue(
            final_state[-1] == 1 and np.all(final_state[:-1] == 0)
        )

    @unittest.skipIf(not TEST_HPC, 'Skipping hpc tests.')
    def test_hpc_default_evolution_operator(self):

        num_vert = 20
        num_steps = num_vert
        entries = [[1, int(num_vert/2), 0],
                   [-1j, int(num_vert/2), 1]]

        seg = hpcoined.Segment(num_vert)
        seg.evolution_operator()
        init_cond = seg.initial_condition(entries)
        seg.time((0, num_steps))
        states = seg.simulate(hpc=False)
        hpc_states = seg.simulate(hpc=True)

        self.assertTrue(
            np.allclose(states, hpc_states, rtol=1e-15, atol=1e-15)
        )
