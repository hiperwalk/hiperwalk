class HamiltonianSimulator(Simulator):
    r"""
    TODO: docs
    """

    def __init__(self, hamiltonian):
        self._hamiltonian = hamiltonian
        self._unitary = None

    def set_matrix(self, *args, **kwargs):
        r"""
        Alias for :meth:`set_hamiltonian`.
        """
        self.set_hamiltonian(*args, **kwargs)
    
    def set_hamiltonian(self, hamiltonian):
        r"""
        TODO
        """
        # TODO: check if hermitian
        self._hamiltonian = hamiltonian
        self._unitary = None

    def get_hamiltonian(self, copy=True):
        r"""
        TODO
        """
        return np.copy(self._hamiltonian) if copy else self._hamiltonian

    def simulate(self, time=None, vector=None, hpc=True):
        r"""
        TODO
        """
        return None
