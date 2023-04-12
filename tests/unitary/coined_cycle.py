import numpy as np
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from test_constants import *
import qwalk.coined as hpcoined
import unittest

class TestCoinedCycle(unittest.TestCase):

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_clockwise_roundabout(self):
        num_steps = 20
        cycle = hpcoined.Cycle(num_steps)

        S = cycle.persistent_shift_operator()
        init_state = cycle.state([(1, 0, 0)])

        final_state = cycle.simulate_walk(S, init_state, num_steps)[0]

        self.assertTrue(np.all(init_state == final_state))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_anticlockwise_roundabout(self):
        num_steps = 20
        cycle = hpcoined.Cycle(num_steps)

        S = cycle.persistent_shift_operator()
        init_state = cycle.state([(1, 0, 1)])

        final_state = cycle.simulate_walk(S, init_state, num_steps)[0]

        self.assertTrue(np.all(init_state == final_state))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_hadamard_evolution_operator(self):
        num_vert = 10
        cycle = hpcoined.Cycle(num_vert)

        U = cycle.evolution_operator()
        init_state = cycle.state([(1, 0, 0)])

        num_steps = num_vert*2
        states = cycle.simulate_walk(U, init_state, num_steps,
                                     save_interval=1)
        states = states.real

        def recursive_expression():
            states = np.zeros((num_steps + 1, init_state.shape[0]))
            states[0] = np.copy(init_state)
            num_arcs = 2*num_vert

            for i in range(1, num_steps + 1):
                for j in range(init_state.shape[0]):
                    if j % 2 == 0:
                        states[i][j] = (states[i-1][(j - 2) % num_arcs]
                                        + states[i-1][(j - 1) % num_arcs]
                                       ) / np.sqrt(2)
                    else:
                        states[i][j] = (states[i-1][(j + 1) % num_arcs]
                                        - states[i-1][(j + 2) % num_arcs]
                                       ) / np.sqrt(2)
            return states


        rec_states = recursive_expression()
        
        self.assertTrue(np.allclose(states, rec_states,
                                    rtol=1e-15, atol=1e-15))

