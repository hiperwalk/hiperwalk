#BDimport numpy as np
#BDfrom sys import path as sys_path
#BDsys_path.append('../')
#BDsys_path.append('../../')
#BDimport os
#BDsys_path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#BDimport hiperwalk as hpw
import unittest
from test_constants import HPC

from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
import hiperwalk as hpw
#from . import _pyhiperblas_interface as nbl

class TestCoinedLine(unittest.TestCase):

    def setUp(self):
        print(f"\n1, em setUp: teste: {self._testMethodName}")
        print(f"\n1, em setUp:  +++  Iniciando")
        hpw.set_hpc(HPC)
        hpw.set_hpc(None)
        hpw.set_hpc("CPU")
        self.num_vert =  3; # 5
        self.num_vert = 11; # 41
        self.line = hpw.Line   (self.num_vert)
        self.qw   = hpw.Coined (self.line)
        print(f"\n2, em setUp:  +++  finalizando ")
        return

    def tearDown(self):
        print(f"\n0, em tearDown: +++  teste: {self._testMethodName}")
        print(f"\n1, em tearDown: +++  iniciando ")
        del self.line
        self.line = None
        del self.qw
        self.qw = None
        print(f"\n2, em tearDown: +++  Finalizando")

    def test04_persistent_shift_left_state_transfer(self):
        # initial state in leftmost vertex
        # final state in rightmost vertex
        print("1, em test04_persistent_shift_left_state_transfer: inicio ")
        self.qw.set_shift('persistent')
        self.qw.set_coin('I')
        self.qw.set_marked([])

        print("1, em test04_per..,  self.num_vert = ", self.num_vert)
        init_state = self.qw.state([[1, (self.num_vert - 1, self.num_vert - 2)]])
        print ("init_state =", init_state)

        num_steps = self.num_vert - 1
        num_steps = self.num_vert - 1 + 2
        print("1.2,em semi-final, num_steps =  self.num_vert - 1 = ", num_steps)
        final_state = self.qw.simulate((num_steps, num_steps + 1), init_state)
        return
        final_stateB = final_state[-1]
        print ("final_stateB =", final_stateB)

        self.assertTrue(
            final_state[0] == 1 and np.all(final_state[1:] == 0)
        )
        print("2, em test04_persistent_shift_left_state_transfer: final ")

    @unittest.skipIf(HPC is None, 'Skipping comparison tests between '
                                  'numpy and PyHiperBlas')

    def Btest02_hpc_default_evolution_operator(self):
        # simulation parameters
        num_steps = self.num_vert // 2
        center = self.num_vert // 2
        entries = [[1, (center, center + 1)],
                   [-1j, (center, center - 1)]]
        init_state = self.qw.state(entries)

        # HPC simulation
        hpc_states = self.qw.simulate((num_steps + 1), init_state)
        # checking if all states are unitary
        probs = self.qw.probability_distribution(hpc_states)
        probs = probs.sum(axis=1)
        # self.assertTrue(
        #     np.allclose(probs, np.ones(probs.shape),
        #                 rtol=1e-15, atol=1e-15)
        # )

        # Non-HPC simulation
        hpw.set_hpc(None)
        states = self.qw.simulate((num_steps + 1), init_state)
        # checking if all states are unitary
        probs = self.qw.probability_distribution(states)
        probs = probs.sum(axis=1)
        # self.assertTrue(
        #     np.allclose(probs, np.ones(probs.shape),
        #                 rtol=1e-15, atol=1e-15)
        # )

        diff = states - hpc_states
        # checking if the obtained states are equivalent
        self.assertTrue(
            np.allclose(states, hpc_states, rtol=1e-15, atol=1e-15)
        )

    def Btest03_set_explicit_coin(self):
        C = self.qw.get_coin()
        self.qw.set_coin(coin=C)
        C2 = self.qw.get_coin()
        self.assertTrue((C - C2).nnz == 0)

    def Btest01_uniform_state(self):
        print("+++   Inicio\n");
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
        print("+++   Final\n");
