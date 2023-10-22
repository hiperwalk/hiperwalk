import numpy as np
from sys import modules as sys_modules
from .._constants import PYNEBLINA_IMPORT_ERROR_MSG
try:
    from sys import path
    path.append('..')
    import _pyneblina_interface as nbl
except ModuleNotFoundError:
    from warnings import warn
    warn(PYNEBLINA_IMPORT_ERROR_MSG)

def pyneblina_imported():
    """
    Expects pyneblina interface to be imported as nbl
    """
    return ('hiperwalk.quantum_walk._pyneblina_interface'
            in sys_modules)

class Simulator():
    r"""
    TODO: docs
    """

    def __init__(self, matrix):
        self.set_matrix(matrix)

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

    def set_matrix(self, matrix):
        r"""
        """
        # TODO: check dimensions
        self._matrix = matrix

    def get_matrix(self, copy=True):
        r"""
        TODO
        """
        return np.copy(self._matrix) if copy else self._matrix

    @staticmethod
    def exponent_to_tuple(exponent):
        r"""
        Clean and format ``exponent`` to ``(start, end, step)`` format.

        See :meth:`simulate` for valid input format options.

        Raises
        ------
        ValueError
            If ``exponent`` is in an invalid input format.
        """
        if not hasattr(exponent, '__iter__'):
            exponent = [exponent]

        if len(exponent) == 1:
            start = end = step = exponent[0]
        elif len(exponent) == 2:
            start = 0
            end = exponent[0]
            step = exponent[1]
        else:
            start = exponent[0]
            end = exponent[1]
            step = exponent[2]

        exponent = [start, end, step]

        if start < 0 or end < 0 or step <= 0:
            raise ValueError(
                "Invalid 'exponent' value."
                + "'start' and 'end' must be non-negative"
                + " and 'step' must be positive."
            )
        if start > end:
            raise ValueError(
                "Invalid `exponent` value."
                + "`start` cannot be larger than `end`."
            )

        return exponent

    ######################################
    ### Auxiliary Simulation functions ###
    ######################################

    def _prepare_engine(self, initial_vector, hpc):
        if self._matrix is None:
            #self._evolution = self.get_evolution(hpc=hpc)
            raise ValueError("Matrix not set.")

        if hpc:
            self._simul_mat = nbl.send_matrix(self._matrix)
            self._simul_vec = nbl.send_vector(initial_vector)

        else:
            self._simul_mat = self._matrix
            self._simul_vec = initial_vector

        dtype = (np.complex128 if (np.iscomplexobj(self._matrix)
                             or np.iscomplexobj(initial_vector))
                 else np.double)

        return dtype

    def _simulate_step(self, step, hpc):
        """
        Apply the simulation evolution operator ``step`` times
        to the simulation vector.
        Simulation vector is then updated.
        """
        if hpc:
            # TODO: request multiple multiplications at once
            #       to neblina-core
            # TODO: check if intermediate vectors are being freed
            for i in range(step):
                self._simul_vec = nbl.multiply_matrix_vector(
                    self._simul_mat, self._simul_vec)
        else:
            for i in range(step):
                self._simul_vec = self._simul_mat @ self._simul_vec

            # TODO: compare with numpy.linalg.matrix_power

    def _save_simul_vec(self, hpc):
        ret = None

        if hpc:
            # TODO: check if vector must be deleted or
            #       if it can be reused via neblina-core commands.
            ret = nbl.retrieve_vector(self._simul_vec)
        else:
            ret = self._simul_vec

        return ret



    def simulate(self, exponent=None, vector=None, hpc=True):
        r"""
        TODO DOCS
        """
        ############################################
        ### Check if simulation was set properly ###
        ############################################
        if exponent is None:
            raise ValueError(
                "``exponent` not specified`. "
                + "Must be an int or tuple of int."
            )

        if vector is None:
            raise ValueError(
                "``vector`` not specified. "
                + "Expected a np.array."
            )

        if len(vector) != self._matrix.shape[1]:
            raise ValueError(
                "Vector has invalid dimension. "
                + "Expected an np.array with length " + str(self._matrix.shape[1])
            )

        ###############################
        ### simulate implemantation ###
        ###############################

        exponent = np.array(Simulator.exponent_to_tuple(exponent))

        if not np.all([e.is_integer() for e in exponent]):
            raise ValueError("`exponent` has non-int entry.")

        start, end, step = exponent

        if hpc and not pyneblina_imported():
            hpc = False

        dtype = self._prepare_engine(vector, hpc)

        # number of vectors to save
        num_vectors = int(end/step) + 1
        num_vectors -= (int((start - 1)/step) + 1) if start > 0 else 0

        saved_vectors = np.zeros(
            (num_vectors, vector.shape[0]), dtype=dtype
        )
        state_index = 0 # index of the state to be saved

        # if save_vector:
        if start == 0:
            saved_vectors[0] = vector.copy()
            state_index += 1
            num_vectors -= 1

        # simulate walk / apply evolution operator
        if start > 0:
            self._simulate_step(start - step, hpc)

        for i in range(num_vectors):
            self._simulate_step(step, hpc)
            saved_vectors[state_index] = self._save_simul_vec(hpc)
            state_index += 1

        # TODO: check if vector is freed from neblina core
        if hpc:
            del self._simul_mat
            del self._simul_vec
        self._simul_mat = None
        self._simul_vec = None

        return saved_vectors
