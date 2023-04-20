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
        cycle.set_evolution_operator(S)
        init_state = cycle.initial_condition([(1, 0, 0)])
        cycle.step(num_steps)

        final_state = cycle.simulate(hpc=False)[0]

        self.assertTrue(np.all(init_state == final_state))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_anticlockwise_roundabout(self):
        num_steps = 20
        cycle = hpcoined.Cycle(num_steps)

        S = cycle.persistent_shift_operator()
        cycle.set_evolution_operator(S)
        init_state = cycle.initial_condition([(1, 0, 1)])
        cycle.time(num_steps)

        final_state = cycle.simulate(hpc=False)[0]

        self.assertTrue(np.all(init_state == final_state))

    @unittest.skipIf(not TEST_NONHPC, 'Skipping nonhpc tests.')
    def test_hadamard_evolution_operator(self):
        num_vert = 10
        cycle = hpcoined.Cycle(num_vert)

        cycle.evolution_operator(hpc=False)
        init_state = cycle.initial_condition([(1, 0, 0)])

        num_steps = num_vert*2
        cycle.time((0, num_steps))
        states = cycle.simulate(hpc=False)
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

        print(states)
        print(rec_states)
        
        self.assertTrue(np.allclose(states, rec_states,
                                    rtol=1e-15, atol=1e-15))

    @unittest.skipIf(not TEST_HPC, 'Skipping hpc tests.')
    def test_hpc_clockwise_roundabout(self):
        num_steps = 20
        cycle = hpcoined.Cycle(num_steps)

        S = cycle.persistent_shift_operator()
        cycle.set_evolution_operator(S)
        init_state = cycle.initial_condition([(1, 0, 0)])
        cycle.step(num_steps)

        final_state = cycle.simulate(hpc=True)[0]

        self.assertTrue(np.all(init_state == final_state))

    @unittest.skipIf(not TEST_HPC, 'Skipping hpc tests.')
    def test_hpc_anticlockwise_roundabout(self):
        num_steps = 20
        cycle = hpcoined.Cycle(num_steps)

        S = cycle.persistent_shift_operator()
        cycle.set_evolution_operator(S)
        init_state = cycle.initial_condition([(1, 0, 1)])
        cycle.time(num_steps)

        final_state = cycle.simulate(hpc=True)[0]

        self.assertTrue(np.all(init_state == final_state))

    @unittest.skipIf(not TEST_HPC, 'Skipping hpc tests.')
    def test_hpc_evolution_operator_matches_nonhpc(self):
        num_vert = 10
        cycle = hpcoined.Cycle(num_vert)

        U = cycle.evolution_operator(hpc=True)
        init_state = cycle.initial_condition([(1, 0, 0)])
        num_steps = num_vert*2
        cycle.step((0, num_steps))

        states = cycle.simulate(hpc=False)
        hpc_states = cycle.simulate(hpc=True)
        
        self.assertTrue(np.allclose(states, hpc_states,
                                    rtol=1e-15, atol=1e-15))
