import numpy as np
import scipy.sparse
import scipy.linalg
from ..base_walk import BaseWalk

class Graph(BaseWalk):
    r"""
    Manage instance of the continuous time quantum walk model
    on unweighted graphs.

    For implemantation details see Notes Section.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on which the quantum occurs
        is going to occur.

        .. todo::
            * Accept other types such as numpy array

    Notes
    -----
    Let :math:`A` be the adjacency matrix of the graph :math:`G(V, E)`.
    :math:`A` is a :math:`|V| \times |V|`-dimensional matrix such that

    .. math::
        A_{i,j} = \begin{cases}
            1, \text{ if } (i,j) \in E(G),\\
            0, \text{ otherwise}
        \end{cases}

    The states of the computational basis are :math:`\ket{i}` for
    :math:`0 \leq i < |V|` where
    :math:`\ket i` is associated with the :math:`i`-th vertex.

    This class can also be used to simulate the evolution of any
    any Hamiltonian.
    Simply pass the disered Hamiltonian instead of the adjacency matrix.
    """

    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)

        self.hilb_dim = self.adj_matrix.shape[0]
        self._hamiltonian = None

        import inspect

        self._valid_kwargs = dict()

        self._valid_kwargs['oracle'] = inspect.getargspec(
            self.oracle
        )[0][1:]

        self._valid_kwargs['hamiltonian'] = inspect.getargspec(
            self.hamiltonian
        )[0][1:]

    def oracle(self, marked_vertices=0):
        r"""
        Creates the oracle matrix.

        The oracle is created and set
        to be used by other methods.

        Parameters
        ----------
        marked_vertices = None, int, list of int, default=0
            Vertices to be marked.
            If ``None``, no vertex is marked and
            the oracle is also set to ``None``
            (which is equivalent to the zero matrix).

        Notes
        -----
        The oracle matrix has format

        .. math::
            \sum_{m \in M} \ket{m}\bra{m}

        where :math:`M` is the set of marked vertices.
        """

        if marked_vertices is None:
            self._oracle = None
            return None

        if not hasattr(marked_vertices, '__iter__'):
            marked_vertices = [marked_vertices]
        self._oracle = np.array(marked_vertices, dtype=int)

        R = np.zeros((self.hilb_dim, self.hilb_dim))
        for m in marked_vertices:
            R[m, m] = 1

        return R

    def hamiltonian(self, gamma=None, laplacian=False, **kwargs):
        r"""
        Creates the Hamiltonian.

        Creates the Hamiltonian based on the previously set oracle.
        If no oracle was set, it is ignored.
        If any valid ``**kwargs`` is sent, an oracle is created and set.
        The new oracle is then used to construct the Hamiltonian.

        Parameters
        ----------
        gamma : float, default : None
            Value to be multiplied by the adjacency matrix or
            the Laplacian.
            If ``None``, an error is raised.

        Returns
        -------
        :class:`numpy.ndarray`

        Raises
        ------
        ValueError
            If ``gamma=None``.
            
        Notes
        -----
        The Hamiltonian is given by

        .. math::
            H = -\gamma B  - \sum_{m \in M} \ket m \bra m

        where :math:`B` is either the adjacency matrix :math:`A`
        or the Laplacian :math:`L = D - A`
        (with :math:`D` being the degree matrix),
        :math:`M` is the set of marked vertices.

        See Also
        ------
        oracle
        """
        if gamma is None:
            raise ValueError(
                "Invalid `gamma` value. It cannot be `None`."
            )

        if laplacian:
            degrees = self.adj_matrix.sum(axis=1)
            #H = np.array(
            #    [[0 if i != j else deg[i] for j in range(self.hilb_dim)]
            #     for i in range(self.hilb_dim)]
            #)
            H = scipy.sparse.diags(degrees)
            del degrees
            H -= self.adj_matrix
            H *= -gamma

        else:
            H = -gamma*self.adj_matrix

        # setting oracle
        oracle_kwargs = self._filter_valid_kwargs(
            kwargs, self._valid_kwargs['oracle'])

        if bool(oracle_kwargs):
            self.oracle(**oracle_kwargs)

        # using previously set oracle
        if self._oracle is not None:
            H -= scipy.sparse.csr_array(
                ([1]*len(self._oracle), (self._oracle, self._oracle)),
                shape=H.shape
            )

        self._hamiltonian = H.todense()
        return self._hamiltonian

    def evolution_operator(self, time=0, hpc=True, **kwargs):
        r"""
        Creates the evolution operator.

        Creates the evolution operator based on the previously
        set Hamiltonian.
        If any valid ``**kwargs`` is passed,
        a new Hamiltonian is created and set.
        The evolution operator is then constructed based on the
        new Hamiltonian.

        Parameters
        ----------
        hpc : bool, default = True
            Whether or not to use neblina hpc functions to
            generate the evolution operator.

        **kwargs :
            Arguments to construct the Hamiltonian.
            See :meth:`hamiltonian` for the list of arguments.
            If no ``**kwrags`` is passed,
            the previously set Hamiltonian is used.
            

        Raises
        ------
        ValueError
            If `time <= 0`.

        AttributeError
            If the Hamiltonian was not set previously and
            no ``**kwargs`` is passed.

        Exception
            If ``**kwargs`` is passed but a valid
            Hamiltonian cannot be created.
            See :meth:`hamiltonian` for details.

        See Also
        ------
        hamiltonian

        Notes
        -----
        The evolution operator is given by

        .. math::
            U = e^{-\text{i}tH}

        where :math:`H` is a Hamiltonian matrix, and
        :math:`t` is the time.

        The evolution operator is constructed by Taylor Series expansion.
        """
        if hpc:
            raise NotImplementedError(
                "No pybnelina function for implementing "
                + "Taylor series expansion available."
            )

        if time <= 0:
            raise ValueError(
                "Expected `time` value greater than 0."
            )

        if bool(kwargs):
            # setting Hamiltonian
            R_kwargs = self._filter_valid_kwargs(
                kwargs, self._valid_kwargs['oracle'])
            H_kwargs = self._filter_valid_kwargs(
                kwargs, self._valid_kwargs['hamiltonian'])

            self.hamiltonian(**H_kwargs, **R_kwargs)

        if self._hamiltonian is None:
            raise AttributeError(
                "Hamiltonian not set. "
                + "Did you forget to call the hamiltonian() method or "
                + "to pass valid **kwargs to evolution_operator()?"
            )
        U = scipy.linalg.expm(-1j*time*self._hamiltonian)
        self._evolution_operator = U
        return U

    def probability_distribution(self, states):
        return None

    def state(self, entries):
        return None

    def simulate(self, time_range=None, hamiltonian=None,
                 initial_condition=None, hpc=True):
        r"""
        Simulate the Continuous Time Quantum Walk Hamiltonian.

        Analogous to the :meth:`BaseWalk.simulate`
        but uses the Hamiltonian to construct the evolution operator.
        The Hamiltonian may be the previously set or
        passed in the arguments.

        Parameters
        ----------
        hamiltonian : :class:`numpy.ndarray` or None
            Hamiltonian matrix to be used for constructing
            the evolution operator.
            If ``None``, uses the previously set Hamiltonian

        Other Parameters
        ----------
        See :meth:`BaseWalk.simulate`.

        Raises
        ------
        ValueError
            If ``time_range=None`` or ``initial_condition=None``,
            or ``hamiltonian`` has invalid Hilbert space dimension.


        Notes
        -----
        It is recommended to call this method with ``hamiltonian=None``
        to guarantee that a valid Hamiltonian was used.
        If the Hamiltonian is passed by the user,
        there is no guarantee that the Hamiltonian is local.

        See Also
        --------
        :meth:`BaseWalk.simulate`
        hamiltonian
        """
        if time_range is None:
            raise ValueError(
                "Invalid `time_range`. Expected a float, 2-tuple, "
                + "or 3-tuple of float".
            )
        if initial_condition is None:
            raise ValueError(
                "`initial_condition` not specified."
            )

        time_range = self._clean_time(time_range)

        if hamiltonian is not None:
            if hamiltonian.shape != (self.hilb_dim, self.hilb_dim):
                raise ValueError(
                    "Hamiltonian has invalid dimensions. "
                    + "Expected Hamiltonian shape: "
                    + str((self.hilb_dim, self.hilb_dim)) + '.'
                )

            prev_R = self._oracle
            prev_H = self._hamiltonian

            self._oracle = None
            self._hamiltonian = hamiltonian
            U = self.evolution_operator(time_range[3])

            self._oracle = prev_R
            self._hamiltonian = prev_H

        return super().simulate_walk(
            self._evolution_operator, initial_condition, num_steps,
            save_interval=save_interval, hpc=hpc
        )
