import neblina
import numpy as np
import scipy.sparse
from warnings import warn
from .._constants import *
from .._constants import __DEBUG__

############################################
# used for automatically stopping the engine
import atexit

__engine_initiated = False

def exit_handler():
    global __engine_initiated
    if __engine_initiated:
        neblina.stop_engine()

atexit.register(exit_handler)
############################################

# "abstract"
class PyNeblinaObject:
    def __init__(self, nbl_obj, shape, is_complex):
        self.nbl_obj = nbl_obj
        self.shape = shape
        self.is_complex = is_complex

class PyNeblinaMatrix(PyNeblinaObject):
    def __init__(self, matrix, shape, is_complex, sparse):
        super().__init__(matrix, shape, is_complex)
        self.sparse = sparse

class PyNeblinaVector(PyNeblinaObject):
    def __init__(self, vector, shape, is_complex):
        super().__init__(vector, shape, is_complex)


def _init_engine():
    r"""
    Initiates neblina-core engine.

    Initiates the engine if it was not previously initiated
    """
    global __engine_initiated
    if not __engine_initiated:
        neblina.init_engine(0)
        __engine_initiated = True

def send_vector(v):
    r"""
    Transfers a vector (v) to Neblina-core, and moves it
    to the device to be used.
    Returns a pointer to this vector
    (needed to call other pyneblina functions).
    by default, a vector with complex entries is expected.
    If the matrix has only real entries, invoke this function by
    TransferVector(v, False);
    this saves half the memory that would be used.
    TODO: is there a way to move the vector to the device directly?
    I think an auxiliary vector is beign created,
    thus twice the memory needed is being used
    """

    _init_engine()
    # TODO: check if complex automatically?
    is_complex = isinstance(v.dtype, complex)

    if not is_complex:
        warn(
            "Real multiplication not implemented. "
            + "Treating entries as complex."
        )
    is_complex = True

    n = v.shape[0]
    # TODO: needs better support from pyneblina to
    # use next instruction (commented).
    # For example: neblina.vector_set works, but in the real case,
    # it should not be needed to pass the imaginary part as argument.
    # In addition,
    # there should be a way to return a vector and automatically
    # convert to an array of float or of complex numbers accordingly.
    # 
    # vec = (neblina.vector_new(n, NEBLINA_COMPLEX)
    #        if is_complex else neblina.vector_new(n, NEBLINA_FLOAT))
    vec = neblina.vector_new(n, NEBLINA_COMPLEX)

    if is_complex:
        for i in range(n):
            neblina.vector_set(vec, i, v[i].real, v[i].imag)
    else:
        for i in range(n):
            # TODO: Pyneblina needs to accept 3 only arguments
            # instead of 4?
            # TODO: check if neblina.vector_set is idetifying
            # the vector type right (i.e. real and not complex)
            neblina.vector_set(vec, i, v[i], 0)

    # suppose that the vector is going to be used
    # immediately after being transferred
    # TODO: check if this is the case
    neblina.move_vector_device(vec)
    return PyNeblinaVector(vec, is_complex, n)

def retrieve_vector(pynbl_vec):
    r"""
    Retrieves vector from the device and converts it to python array.
    By default, it is supposed that the vector is not going to be used in
    other pyneblina calculations,
    thus it is going to be deleted to free memory.
    TODO: get vector dimension(vdim) automatically
    """

    # if a vector is being retrieved.
    # the engine should have been already initiated
    if __DEBUG__:
        global __engine_initiated
        if not __engine_initiated: raise AssertionError

    nbl_vec = neblina.move_vector_host(pynbl_vec.nbl_obj)

    if not pynbl_vec.is_complex:
        raise NotImplementedError("Cannot retrieve real-only vectors.")
    py_vec = np.array(
                [neblina.vector_get(nbl_vec, 2*i)
                 + 1j*neblina.vector_get(nbl_vec, 2*i + 1)
                 for i in range(pynbl_vec.shape)]
            )

    # TODO: check if vector is being deleted (or not)

    return py_vec
        
def _send_sparse_matrix(M, is_complex):
    r"""
    Transfers a sparse Matrix (M) stored in csr format to Neblina-core and
    moves it to the device (ready to be used).
    By default, a matrix with complex elements is expected.
    If the matrix has only real elements, invoke this function by
    TransferSparseMatrix(M, False);
    this saves half the memory that would be used.
    TODO: Add tests
      - Transfer and check real Matrix
      - Transfer and check complex Matrix
    TODO: isn't there a way for neblina-core to use the csr matrix directly?
      In order to avoid double memory usage
    """
    
    _init_engine()

    # TODO: check if complex automatically?
    n = M.shape[0]

    # creates neblina sparse matrix structure
    # TODO: needs better support from pyneblina to
    #   use next instruction (commented).
    #   For example: neblina.sparse_matrix_set works, but in the real case,
    #   it should not be needed to pass the imaginary part as argument.
    #   In addition, there should be a way to
    #   return the matrix and automatically
    #   convert to a matrix of float or of complex numbers accordingly.
    # smat = neblina.sparse_matrix_new(n, n, NEBLINA_COMPLEX)
    #     if is_complex else neblina.sparse_matrix_new(n, n, NEBLINA_FLOAT)
    smat = neblina.sparse_matrix_new(n, n, NEBLINA_COMPLEX)
    
    # inserts elements into neblina sparse matrix
    row = 0
    next_row_ind = M.indptr[1]
    j = 2
    for i in range(len(M.data)):
        while i == next_row_ind:
            row += 1
            next_row_ind = M.indptr[j]
            j += 1
            
        col = M.indices[i]
        if is_complex:
            neblina.sparse_matrix_set(smat, row, col, M[row, col].real,
                              M[row, col].imag)
        else:
            # TODO: Pynebliena needs to accept 4 arguments instead of 5?
            # TODO: check if smatrix_set_real_value is beign called
            # instead of smatrix_set_complex_value
            neblina.sparse_matrix_set(smat, row, col, M[row, col], 0) 

    neblina.sparse_matrix_pack(smat) # TODO: is this needed?
    neblina.move_sparse_matrix_device(smat)

    return PyNeblinaMatrix(smat, M.shape, is_complex, True)

def _send_dense_matrix(M, is_complex):
    _init_engine()

    num_rows, num_cols = M.shape
    mat = (neblina.matrix_new(num_rows, num_cols, NEBLINA_COMPLEX)
           if is_complex
           else neblina.matrix_new(num_rows, num_cols, NEBLINA_FLOAT))
    
    # inserts elements into neblina matrix
    # TODO: Check if there really is a difference between real and complex
    # TODO: Check if default value is zero so we can jump setting element. 

    for i in range(num_rows):
        for j in range(num_cols):
            neblina.matrix_set(mat, i, j, M[i, j].real, M[i, j].imag)

    neblina.move_matrix_device(mat)

    return PyNeblinaMatrix(mat, M.shape, is_complex, False)

def send_matrix(M):
    is_complex = isinstance(M.dtype, complex)

    if not is_complex:
        warn(
            "Real multiplication not implemented. "
            + "Treating entries as complex."
        )
    is_complex = True

    if scipy.sparse.issparse(M):
        return _send_sparse_matrix(M, is_complex)

    return _send_dense_matrix(M, is_complex)

def retrieve_matrix(pynbl_mat):

    if pynbl_mat.sparse:
        raise NotImplementedError(
            "Cannot retrieve sparse matrix."
        )

    nbl_mat = neblina.move_matrix_host(pynbl_mat.nbl_obj)

    # TODO: Check if using default numpy datatype.
    py_mat = np.zeros(pynbl_mat.shape, dtype=(
        complex if pynbl_mat.is_complex else float
    ))

    num_rows, num_cols = pynbl_mat.shape
    # TODO: Not vectorized. Implement with list comprehension. 
    #       This may require the double memory usage. 
    #       Check memory usage before choosing which method to use.
    for i in range(num_rows):
        for j in range(num_cols):
            if pynbl_mat.is_complex:
                py_mat[i,j] = (
                      neblina.matrix_get(nbl_mat, 2*i, 2*j)
                    + neblina.matrix_get(nbl_mat, 2*i, 2*j + 1)*1j
                )
            else:
                py_mat[i,j] = neblina.matrix_get(nbl_mat, i, j)

    return py_mat

def multiply_matrix_vector(pynbl_mat, pynbl_vec):
    """
    Request matrix multiplication to neblina.

    Multiplies the matrix by the vector, i.e. ``smat @ vec``.

    Parameters
    ----------
    mat : :class:`PyNeblinaMatrix`
        neblina matrix object
    vec : :class:`PyNeblinaVector`
        neblina vector object

    Returns
    -------
    Neblina vector object resulted from matrix multiplication.

    See Also
    --------
    send_sparse_matrix : returns a neblina sparse matrix object
    send_vector : returns a neblina vector object
    """
    # if a matrix-vector operation is being requested,
    # the engine should have been already initiated
    if __DEBUG__:
        global __engine_initiated
        if not __engine_initiated: raise AssertionError

    if pynbl_mat.sparse:
        nbl_vec = neblina.sparse_matvec_mul(pynbl_vec.nbl_obj,
                                            pynbl_mat.nbl_obj)
    else:
        nbl_vec = neblina.matvec_mul(pynbl_vec.nbl_obj,
                                     pynbl_mat.nbl_obj)

    new_vec = PyNeblinaVector(
        nbl_vec, pynbl_mat.shape[0],
        pynbl_mat.is_complex or pynbl_vec.is_complex
    )

    return new_vec

def multiply_matrices(pynbl_A, pynbl_B):

    if pynbl_A.shape[1] != pynbl_B.shape[0]:
        raise ValueError("Matrices dimensions do not match.")

    if pynbl_A.sparse or pynbl_B.sparse:
        raise NotImplementedError(
            "No sparse matrix multiplication implemented."
        )

    A = pynbl_A.nbl_obj
    B = pynbl_B.nbl_obj
    C = neblina.mat_mul(A, B)

    pynbl_C = PyNeblinaMatrix(
        C, (pynbl_A.shape[0], pynbl_B.shape[1]),
        pynbl_A.is_complex or pynbl_B.is_complex,
        False
    )

    return pynbl_C

def matrix_power_series(A, n):
    r"""
    Computes the following power series.
    A must be a dense numpy matrix.

    I + A + A^2/2 + A^3/3! + A^4/4! + ... + A^n/n!
    """
    if scipy.sparse.issparse(A):
        A = np.array(A.todense())
        warn(
            "Sparse matrix multiplication not implemented. "
            + "Converting to dense."
        )

    _init_engine()
    pynbl_A = send_matrix(A)
    pynbl_Term = send_matrix(np.eye(A.shape[0], dtype=A.dtype))
    pynbl_M = send_matrix(np.eye(A.shape[0], dtype=A.dtype))

    for i in range(1, n+1):
        pynbl_Term.nbl_obj = neblina.mat_mul(
                pynbl_Term.nbl_obj, pynbl_A.nbl_obj
        )
        pynbl_Term.nbl_obj = neblina.scalar_mat_mul(
                1/i, pynbl_Term.nbl_obj
        )
        pynbl_M.nbl_obj = neblina.mat_add(
                pynbl_M.nbl_obj, pynbl_Term.nbl_obj
        )

    return pynbl_M
