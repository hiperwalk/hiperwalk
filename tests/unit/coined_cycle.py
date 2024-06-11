import numpy as np
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from test_constants import *
import hiperwalk as hpw
import unittest
from test_constants import HPC

class TestCoinedCycle(unittest.TestCase):

    def setUp(self):
        hpw.set_hpc(HPC)
        self.num_vert = 10
        self.cycle = hpw.Cycle(self.num_vert)
        self.qw = hpw.Coined(self.cycle)


    def test_clockwise_roundabout(self):
        self.qw.set_shift('persistent')
        self.qw.set_coin('I')
        self.qw.set_marked([])

        init_state = self.qw.state([[1, (0, 1)]])

        num_steps = self.num_vert
        final_state = self.qw.simulate((num_steps, num_steps + 1),
                                       init_state)
        final_state = final_state[0]

        self.assertTrue(np.all(init_state == final_state))

    def test_anticlockwise_roundabout(self):
        self.qw.set_shift('persistent')
        self.qw.set_coin('I')
        self.qw.set_marked([])

        init_state = self.qw.state([[1, (0, self.num_vert - 1)]])

        num_steps = self.num_vert
        final_state = self.qw.simulate((num_steps, num_steps + 1),
                                       init_state)
        final_state = final_state[0]

        self.assertTrue(np.all(init_state == final_state))

    def test_hadamard_evolution_operator(self):
        init_state = self.qw.state([(1, (0, 1))])

        num_steps = 2*self.num_vert
        states = self.qw.simulate(num_steps + 1, init_state)
        states = states.real

        def recursive_expression():
            states = np.zeros((num_steps + 1, init_state.shape[0]))
            states[0] = np.copy(init_state)
            num_arcs = 2*self.num_vert

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

    @unittest.skipIf(HPC is None, 'Skipping comparison tests between '
                                  'numpy and PyHiperBlas')
    def test_hpc_evolution_operator_matches_nonhpc(self):
        init_state = self.qw.ket((0, 1))
        num_steps = 2*self.num_vert

        hpw.set_hpc(None)
        states = self.qw.simulate(num_steps + 1, init_state)

        hpw.set_hpc(HPC)
        hpc_states = self.qw.simulate(num_steps + 1, init_state)
        
        self.assertTrue(np.allclose(states, hpc_states,
                                    rtol=1e-15, atol=1e-15))

    def test_set_explicit_coin(self):
        C = self.qw.get_coin()
        self.qw.set_coin(coin=C)
        C2 = self.qw.get_coin()
        self.assertTrue((C - C2).nnz == 0)
