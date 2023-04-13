from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse
from constants import DEBUG

class BaseWalk(ABC):
    """
    Base class for Quantum Walks.

    Base methods and attributes used for implementing
    specific Quantum Walk models.

    .. todo::
        The following methods must be overwritten.

    Parameters
    ----------
    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on
        which the quantum walk occurs.

    Attributes
    ----------
    hilb_dim : int, default=0
        Hilbert Space dimension.
        It must be updated by the subclass' ``__init__``.

    adj_matrix : :class:`scipy.sparse.csr_array`
        Adjacency matrix of the graph on
        which the quantum walk occurs.

    Raises
    ------
    TypeError
        if ``adj_matrix`` is not an instance of
        :class:`scipy.sparse.csr_array`.
    """

    @abstractmethod
    def __init__(self, adj_matrix):
        self._initial_condition = None
        self._evolution_operator = None
        self._steps = 0

        ##############################
        ### Simulation attributes. ###
        ##############################
        # Matrix object used during simulation.
        # It may by a scipy matrix or a neblina matrix.
        # Should be different from None during simulation only.
        self._simul_mat = None
        # Vector object used during simulation.
        # Should be different from None during simulation only.
        self._simul_vec = None

        # TODO: create sparse matrix from graph or dense adjacency matrix
        if isinstance(adj_matrix, scipy.sparse.csr_array):
            self.adj_matrix = adj_matrix
        else:
            raise TypeError(
                "Invalid 'adj_matrix' type."
                + " Expected 'scipy.sparse.csr_array',"
                + " but received " + str(type(adj_matrix)) + '.'
            )

        self.hilb_dim = 0

    def uniform_state(self):
        r"""
        Generate the uniform state.

        The state is constructed based on the ``hilb_dim`` attribute.
        The uniform initial condition is the state where
        all entries have the same amplitude.

        Returns
        -------
        :obj:`numpy.ndarray`

        Notes
        -----
        The uniform initial condition is the state

        .. math::

            \ket{d} = \frac{1}{\sqrt{N}} \sum_{i = 0}^{N - 1} \ket{i}

        where :math:`N` is the dimension of the Hilbert space.
        """
        return (np.ones(self.hilb_dim, dtype=float)
                / np.sqrt(self.hilb_dim))

    def uniform_initial_condition(self):
        """
        Alias for :obj:`uniform_state`.
        """
        return self.uniform_state()

    @abstractmethod
    def oracle(self, vertices):
        r"""
        Create the oracle that marks the given vertices.

        Parameters
        ----------
        vertices : array_like
            ID(s) of the vertex (vertices) to be marked.

        Returns
        -------
        :class:`scipy.sparse.csr_array`
        """
        return None

    @abstractmethod
    def evolution_operator(self, hpc=False, **kwargs):
        """
        Create the standard evolution operator.

        Parameters
        ----------
        hpc : bool, default=False
            Whether or not evolution operator should be
            constructed using nelina's high-performance computating.

        **kwargs : dict, optional
            Additional arguments for constructing the evolution operator

        Returns
        -------
        U_w : :class:`scipy.sparse.csr_array`
            The evolution operator.
        """
        return None

    @abstractmethod
    def search_evolution_operator(self, vertices, hpc=False,
                                  **kwargs):
        """
        Create the search evolution operator.

        Parameters
        ----------
        vertices : array_like
            The marked vertex (vertices) IDs.
            See :obj:`oracle`'s ``vertices`` parameter.

        hpc : bool, default=False
            Whether or not evolution operator should be
            constructed using nelina's high-performance computating.

        **kwargs : dict, optional
            Additional arguments for constructing the evolution operator

        Returns
        -------
        U : :class:`scipy.sparse.csr_array`
            The search evolution operator.

        See Also
        --------
        evolution_operator
        oracle
        """
        return None

    @staticmethod
    def _elementwise_probability(elem):
        # This is more efficient than:
        # (np.conj(elem) * elem).real
        # elem.real**2 + elem.imag**2
        return elem.real*elem.real + elem.imag*elem.imag

    @abstractmethod
    def probability_distribution(self, states):
        """
        Compute the probability distribution of given states.

        The probability of the walker being found on each vertex
        for the given states.

        Parameters
        ----------
        states : :class:`numpy.ndarray`
            The states used to compute the probabilities.

        Returns
        -------
        probabilities : :class:`numpy.ndarray`

        See Also
        --------
        simulate_walk
        """
        return None

    def _clean_steps(self, steps):
        r"""
        Clean and format ``steps`` to ``(start, end, step)`` format.

        See :meth:`simulate_walk` for valid input format options.

        Raises
        ------
        ValueError
            If ``steps`` is in an invalid input format.

        See Also
        --------
        simulate_walk
        """

        if not hasattr(steps, '__iter__'):
            steps = [steps]

        start = steps[0]
        end = steps[1] if len(steps) >= 2 else start
        step = steps[2] if len(steps) >= 3 else 1

        steps = (start, end, step)

        if start < 0 or end < 0 or step <= 0:
            raise ValueError(
                "Invalid 'steps' value."
                + "'start' and 'end' must be non-negative"
                + " and 'step' must be positive."
            )

        if (end - start)%step != 0:
            raise ValueError(
                "Invalid 'steps' value."
                + "'start' and 'end' are not a multiple of "
                + "'step' steps apart"
            )

        return steps


    def simulate_walk(self, evolution_operator, initial_condition,
                      steps, hpc=False):
        r"""
        Simulates quantum walk by applying the
        ``evolution_operator`` to the ``initial_coidition``
        multiple times.

        The maximum number of applications is described by ``steps``.

        Parameters
        ----------
        evolution_operator
            Operator that describes the quantum walk.

        initial_condition
            The initial state.

        steps : int, 2-tuple or 3-tuple
            Describes at which steps the state must be saved.
            It can be specified in three different ways.
            
            * ``end``
                The evolution operator is applied
                ``end`` times. Only the final state is saved.

            * ``(start, end)``
                Saves each state from the
                ``start``-th to the ``end``-th application
                of the evolution operator.
                That is, ``[start, start + 1, ..., end - 1, end]``.

            * ``(start, end, step)``
                Saves every state from the
                ``start``-th to the ``end``-th application
                of the evolution operator separated by
                ``step`` applications.
                That is, ``[start, start + step, ..., end - step, end]``.
            
        hpc : bool, default=False
            Whether or not to use neblina's high-performance computing
            to perform matrix multiplications.
            If ``hpc=False`` uses python.

        Returns
        -------
        states : :class:`numpy.ndarray`.
            States saved during simulation where
            ``states[i]`` corresponds to the ``i``-th saved state.

        Raises
        ------
        ValueError
            If ``steps=(start, end, step)`` and
            ``end`` cannot be reached from ``start`` after a
            multiple of ``step`` applications.
            In other words, if
            ``end - start`` is not a multiple of ``step``.
            
            It is also raised if any of the following occurs.
            
            * ``start < 0``,
            * ``end < 0``,
            * ``step <= 0``.


        Notes
        -----
        The parameters ``evolution_operator``, ``initial_condition``,
        and ``steps`` are saved as attributes for possible later usage.

        .. todo::
            Implement assertion of arguments.
            For example: check if evolution operator is unitary and
            if locality is respected.

        Examples
        --------
        If ``steps=(0, 12, 3)``, the returned saved states are:
        the initial state, the intermediate states (3, 6, and 9),
        and the final state (12).

        >>> qw.simulate_walk(U, psi0, (0, 12, 3))
        """
        ###########################
        ### Auxiliary functions ###
        ###########################

        def __save_simulation_parameters(self, evolution_operator,
                         initial_condition, steps):
            self._evolution_operator = evolution_operator
            self._initial_condition = initial_condition
            self._steps = steps

        def __prepare_engine(self):
            if DEBUG:
                print("Preparing engine")

            if hpc:
                print("send sparse")
                self._simul_mat = nbl.send_sparse_matrix(
                    self._evolution_operator)
                print("send vector")
                self._simul_vec = nbl.send_vector(
                    self._initial_condition)

            else:
                self._simul_mat = self._evolution_operator
                self._simul_vec = self._initial_condition

            if DEBUG:
                print("Done\n")

        def __simulate_steps(self, num_steps):
            """
            Apply the simulation evolution operator ``num_steps`` times
            to the simulation vector.
            Simulation vector is then updated.
            """
            if DEBUG:
                print("Simulating steps")

            if hpc:
                # TODO: request multiple multiplications at once
                #       to neblina-core
                # TODO: check if intermediate states are being freed
                for i in range(num_steps):
                    self._simul_vec = nbl.multiply_sparse_matrix_vector(
                        self._simul_mat, self._simul_vec)
            else:
                for i in range(num_steps):
                    self._simul_vec = self._simul_mat @ self._simul_vec

                # TODO: compare with numpy.linalg.matrix_power

            if DEBUG:
                print("Done\n")

        def __save_simul_vec(self):
            if DEBUG:
                print("Saving simulated vec")

            ret = None

            if hpc:
                # TODO: check if vector must be deleted or
                #       if it can be reused via neblina-core commands.
                ret = nbl.retrieve_vector(
                    self._simul_vec, self._initial_condition.shape[0],
                    delete_vector=True
                )
            else:
                ret = self._simul_vec

            if DEBUG:
                print("Done\n")

            return ret


        ####################################
        ### simulate_walk implemantation ###
        ####################################

        start, end, step = self._clean_steps(steps)
        
        __save_simulation_parameters(self, evolution_operator,
                                     initial_condition, steps)

        if hpc:
            from . import _pyneblina_interface as nbl
        __prepare_engine(self)

        # number of states to save
        num_states = int((end - start)/step) + 1

        # create saved states matrix
        dtype = (self._initial_condition.dtype if
            self._initial_condition.dtype == self._evolution_operator.dtype
            else complex
        )
        saved_states = np.zeros(
            (num_states, self._initial_condition.shape[0]), dtype=dtype
        )
        state_index = 0 # index of the state to be saved

        # if save_initial_state:
        if start == 0:
            saved_states[0] = self._initial_condition
            state_index += 1
            num_states -= 1

        # simulate walk / apply evolution operator
        if start > 0:
            __simulate_steps(self, start - step)

        for i in range(num_states):
            __simulate_steps(self, step)
            saved_states[state_index] = __save_simul_vec(self)
            state_index += 1

        # TODO: free vector from neblina core
        self._simul_mat = None
        self._simul_vec = None

        return saved_states
