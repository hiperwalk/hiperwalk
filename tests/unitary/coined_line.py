import numpy as np
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
import qwalk.coined as hpcoined
import unittest

class TestCoinedLine(unittest.TestCase):

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
