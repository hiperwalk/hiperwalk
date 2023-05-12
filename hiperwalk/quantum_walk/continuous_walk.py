import numpy as np
import scipy.sparse
import scipy.linalg
from .quantum_walk import QuantumWalk
from .._constants import PYNEBLINA_IMPORT_ERROR_MSG
from warnings import warn

class ContinuousWalk(QuantumWalk):
    r"""
    Manage instance of the continuous time quantum walk model
    on unweighted graphs.

    For implemantation details see Notes Section.

    Parameters
    ----------
    graph :
        Graph on which the quantum walk occurs.
        There are three two types acceptable.

        :class:`hiperwalk.graph.Graph` :
            The graph itself.

        :class:`class:scipy.sparse.csr_array`:
            The graph adjacency matrix.

        .. todo::
            * Accept other types such as numpy array



    adjacency : :class:`scipy.sparse.csr_array`
        .. deprecated ::
            This parameter is deprecated and will be removed in
            future versions.
            Use ``graph`` instead.

        Adjacency matrix of the graph on which the quantum occurs
        is going to occur.

    **kwargs : optional
        Optional arguments for setting the non-default evolution operator.
        See :meth:`set_evolution`.

    See Also
    --------
    set_evolution

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

    def __init__(self, graph=None, adjacency=None, **kwargs):

        super().__init__(graph=graph, adjacency=adjacency)

        self.hilb_dim = self.adj_matrix.shape[0]
        self._hamiltonian = None

        import inspect

        self._valid_kwargs = dict()

        self._valid_kwargs['oracle'] = self._get_valid_kwargs(
            self.oracle)
        self._valid_kwargs['hamiltonian'] = self._get_valid_kwargs(
            self.hamiltonian)

    def set_laplacian(self, laplacian=False):
        r"""
        Set whether or not to use Laplacian for the Hamiltonian.

        Parameters
        ----------
        laplacian : bool, default=False
            If ``False`` uses adjacency matrix.
            If ``True`` uses Laplacian matrix.
        """
        self._laplacian = bool(laplacian)

    def get_laplacian(self):
        r"""
        Get whether the laplacian is used for the Hamiltonian.

        Returns
        -------
        bool
        """
        return self._laplacian

    def set_gamma(self, gamma=1):
        r"""
        Set gamma used for the Hamiltonian.

        Parameters
        ----------
        gamma : float, default = 1
            Gamma value.
        """
        if gamma.imag != 0:
            raise TypeError("Value of 'gamma' is not float.")

        self._gamma = gamma

    def get_gamma(self):
        r"""
        Get the gamma used for the Hamiltonian.

        Returns
        -------
        float
        """
        return self._gamma

    def oracle(self, marked_vertices=0):
        r"""
        Creates the oracle matrix.

        The oracle is created and set
        to be used by other methods.
        For coherence, the previously set hamiltonian and
        evolution operator are unset.

        Parameters
        ----------
        marked_vertices = None, int, list of int, default=0
            Vertices to be marked.
            If ``None``, no vertex is marked and
            the oracle is also set to ``None``
            (which is equivalent to the zero matrix).

        Returns
        -------
        :class:`scipy.sparse.csr_array`

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
        self._oracle = scipy.sparse.csr_array(
            ([1]*len(marked_vertices),
                (marked_vertices, marked_vertices)
            ),
            shape=(self.hilb_dim, self.hilb_dim)
        )
    # since the oracle was set,
        # the previous hamiltonian and evolution operator
        # are probably not coherent with the oracle
        self._hamiltonian = None
        self._evolution_operator = None

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

        For coherence, the previously set evolution operator is unset.

        Parameters
        ----------
        gamma : float, default : None
            Value to be multiplied by the adjacency matrix or
            the Laplacian.
            If ``None``, an error is raised.

        laplacian : bool, default : False
            Whether to construct the Hamiltonian using the
            adjacency matrix or the Laplacian.

        **kwargs :
            Additional arguments.
            Used for determining the marked vertices.
            See :meth:`oracle` for valid keywords and values.

        Returns
        -------
        :class:`scipy.sparse.csr_array`

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
        --------
        oracle
        """
        if gamma is None:
            raise ValueError(
                "Invalid `gamma` value. It cannot be `None`."
            )

        if laplacian:
            degrees = self.adj_matrix.sum(axis=1)
            H = scipy.sparse.diags(degrees, format="csr")
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
            H -= self._oracle

        self._hamiltonian = H
        # since the hamiltonian was changed,
        # the previous evolution operator may not be coherent.
        self._evolution_operator = None
        return H

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
        --------
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

        if hpc and not self._pyneblina_imported():
            try:
                from .. import _pyneblina_interface as nbl
            except ModuleNotFoundError:
                warn(PYNEBLINA_IMPORT_ERROR_MSG)
                hpc = False

        if hpc:
            # determining the number of terms in power series
            max_val = np.max(np.abs(self._hamiltonian))
            if max_val*time <= 1:
                nbl_U = nbl.matrix_power_series(
                        -1j*time*self._hamiltonian, 30)

            else:
                # if the order of magnitude is very large,
                # float point errors may occur
                if ((isinstance(time, int) or time.is_integer())
                    and max_val <= 1
                ):
                    new_time = 1
                    num_mult = time - 1
                else:
                    new_time = max_val*time
                    order = np.ceil(np.math.log(new_time, 20))
                    new_time /= 10**order
                    num_mult = int(np.round(time/new_time)) - 1
                    warn("Result is approximated. It is recommended to "
                         + "choose a small time interval and performing "
                         + "multiple matrix multiplications to "
                         + "mitigate uounding errors.")

                new_nbl_U = nbl.matrix_power_series(
                        -1j*new_time*self._hamiltonian, 20)
                nbl_U = nbl.multiply_matrices(new_nbl_U, new_nbl_U)
                for i in range(num_mult - 1):
                    nbl_U = nbl.multiply_matrices(nbl_U, new_nbl_U)

            U = nbl.retrieve_matrix(nbl_U)

        else:
            U = scipy.linalg.expm(-1j*time*self._hamiltonian.todense())

        self._evolution_operator = U
        return U

    def simulate(self, time_range=None, initial_condition=None,
                 hamiltonian=None, hpc=True):
        r"""
        Simulate the Continuous Time Quantum Walk Hamiltonian.

        Analogous to the :meth:`BaseWalk.simulate`
        but uses the Hamiltonian to construct the evolution operator.
        The Hamiltonian may be the previously set or
        passed in the arguments.

        Parameters
        ----------
        time_range : float or tuple of floats
            Analogous to the parameters of :meth:`BaseWalk.simulate`,
            but accepts float inputs.
            ``step`` is used to construct the evolution operator.
            The states in the interval
            ***[* ``start/step``, ``end/step`` **]** are saved.
            The values that describe this interval are
            rounded up if the decimal part is greater than ``1 - 1e-5``,
            and rounded down otherwise.

        hamiltonian : :class:`numpy.ndarray` or None
            Hamiltonian matrix to be used for constructing
            the evolution operator.
            If ``None``, uses the previously set Hamiltonian

        Other Parameters
        ----------------
        initial_condition :
            See :meth:`qwalk.BaseWalk.simulate`.
        hpc :
            See :meth:`qwalk.BaseWalk.simulate`.


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
        :meth:`qwalk.BaseWalk.simulate`
        hamiltonian
        """
        if time_range is None:
            raise ValueError(
                "Invalid `time_range`. Expected a float, 2-tuple, "
                + "or 3-tuple of float."
            )

        if initial_condition is None:
            raise ValueError(
                "`initial_condition` not specified."
            )

        time_range = np.array(self._time_range_to_tuple(time_range))

        if hamiltonian is not None:
            if hamiltonian.shape != (self.hilb_dim, self.hilb_dim):
                raise ValueError(
                    "Hamiltonian has invalid dimensions. "
                    + "Expected Hamiltonian shape: "
                    + str((self.hilb_dim, self.hilb_dim)) + '.'
                )

            prev_R = self._oracle
            prev_H = self._hamiltonian
            prev_U = self._evolution_operator

            self._oracle = None
            self._hamiltonian = hamiltonian
            U = self.evolution_operator(time_range[2], hpc=hpc)

            self._oracle = prev_R
            self._hamiltonian = prev_H
            self._evolution_operator = prev_U
        else:
            self.evolution_operator(time_range[2], hpc=hpc)
            U = None # use the set evolution operator

        # cleaning time_range to int
        tol = 1e-5
        time_range = [int(val/time_range[2])
                      if int(val/time_range[2])
                         <= np.ceil(val/time_range[2]) - tol
                      else int(np.ceil(val/time_range[2]))
                      for val in time_range]

        states = super().simulate(time_range, initial_condition, U,  hpc)
        return states
