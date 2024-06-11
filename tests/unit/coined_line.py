import numpy as np
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
import hiperwalk as hpw
import unittest
from test_constants import HPC

class TestCoinedLine(unittest.TestCase):

    def setUp(self):
        hpw.set_hpc(HPC)
        self.num_vert = 41
        self.line = hpw.Line(self.num_vert)
        self.qw = hpw.Coined(self.line)

    def test_persistent_shift_right_state_transfer(self):
        # initial state in leftmost vertex
        # final state in rightmost vertex
        print("test_persistent_shift_right_state_transfer")
        self.qw.set_shift('persistent')
        self.qw.set_coin('I')
        self.qw.set_marked([])

        init_state = self.qw.state([[1, (0, 1)]])

        num_steps = self.num_vert - 1
        final_state = self.qw.simulate((num_steps, num_steps + 1),
                                       init_state)
        final_state = final_state[0]

        self.assertTrue(
            final_state[-1] == 1 and np.all(final_state[:-1] == 0)
        )

    def test_persistent_shift_left_state_transfer(self):
        print("test_persistent_shift_left_state_transfer")
        # initial state in leftmost vertex
        # final state in rightmost vertex
        self.qw.set_shift('persistent')
        self.qw.set_coin('I')
        self.qw.set_marked([])

        init_state = self.qw.state(
            [[1, (self.num_vert - 1, self.num_vert - 2)]])

        num_steps = self.num_vert - 1
        final_state = self.qw.simulate((num_steps, num_steps + 1),
                                       init_state)
        final_state = final_state[0]

        self.assertTrue(
            final_state[0] == 1 and np.all(final_state[1:] == 0)
        )

    @unittest.skipIf(HPC is None, 'Skipping comparison tests between '
                                  'numpy and PyHiperBlas')
    def test_hpc_default_evolution_operator(self):
        print("test_hpc_default_evolution_operator")

        num_steps = self.num_vert // 2
        center = self.num_vert // 2
        entries = [[1, (center, center + 1)],
                   [-1j, (center, center - 1)]]
        init_state = self.qw.state(entries)

        hpw.set_hpc(None)
        states = self.qw.simulate((num_steps + 1), init_state)
        hpw.set_hpc(HPC)
        hpc_states = self.qw.simulate((num_steps + 1), init_state)

        self.assertTrue(
            np.allclose(states, hpc_states, rtol=1e-15, atol=1e-15)
        )

    def test_set_explicit_coin(self):
        print("test_set_explicit_coin")
        C = self.qw.get_coin()
        self.qw.set_coin(coin=C)
        C2 = self.qw.get_coin()
        self.assertTrue((C - C2).nnz == 0)

    def test_uniform_state(self):
        print("test_uniform_state")
        state = self.qw.uniform_state()
        num_arcs = self.qw._graph.number_of_arcs()
        
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (num_arcs, ))
        self.assertTrue(np.allclose(
            state,
            1/np.sqrt(num_arcs)*np.ones(num_arcs)
        ))

        # superposition of given arcs
        even_arcs = np.arange(0, num_arcs, 2)
        state = self.qw.uniform_state(arcs=even_arcs)
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (num_arcs, ))
        self.assertTrue(np.allclose(
            state,
            [1/np.sqrt(len(even_arcs)) if a % 2 == 0 else 0
             for a in range(num_arcs)]
        ))
        odd_arcs = np.arange(1, num_arcs, 2)
        state = self.qw.uniform_state(arcs=odd_arcs)
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (num_arcs, ))
        self.assertTrue(np.allclose(
            state,
            [1/np.sqrt(len(odd_arcs)) if a % 2 == 1 else 0
             for a in range(num_arcs)]
        ))

        # test if all vertices == uniform superposition
        self.assertTrue(np.allclose(
            self.qw.uniform_state(),
            self.qw.uniform_state(vertices=np.arange(self.num_vert))
        ))
        # test if all arcs == uniform superposition
        self.assertTrue(np.allclose(
            self.qw.uniform_state(),
            self.qw.uniform_state(arcs=np.arange(num_arcs))
        ))
        # test if all vertices and all arcs == uniform_superposition
        self.assertTrue(np.allclose(
            self.qw.uniform_state(),
            self.qw.uniform_state(vertices=np.arange(self.num_vert),
                                  arcs=np.arange(num_arcs))
        ))

        # uniform superposition of all arcs
        # except odd arcs with tail in odd vertices
        even_verts = np.arange(0, self.num_vert, 2)
        state = self.qw.uniform_state(vertices=even_verts,
                                      arcs=even_arcs)
        state2 = [0 if (a % 2 == 1 and
                        self.qw._graph.arc(a)[0] % 2 == 1)
                  else 1
                  for a in range(num_arcs)]
        state2 = state2 / np.sqrt(np.sum(state2))
        self.assertTrue(np.allclose(state, state2))
