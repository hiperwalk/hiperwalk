try:
    import neblina
except ModuleNotFoundError:
    pass
import numpy as np
import scipy.sparse
from warnings import warn
from .._constants import *

############################################
# used for automatically stopping the engine
import atexit

__engine_initiated = False
__hpc_type = None

def set_hpc(hpc):
    r"""
    Indicate which HPC platform is going to be used.

    After executing the ``set_hpc`` command,
    all subsequent hiperwalk commands will
    use the designated HPC platform.

    Parameters
    ----------
    hpc : {None, 'cpu', 'gpu'}
        Indicates whether to utilize HPC
        for matrix multiplication using CPU or GPU.
        If ``hpc=None``, it will use standalone Python.
    """
    new_hpc = hpc

    if hpc is not None:
        hpc = hpc.lower()
        hpc = hpc.strip()

        if hpc == 'cpu':
            new_hpc = 0
        elif hpc == 'gpu':
            new_hpc = 1
        else:
            raise ValueError(
                    'Unexpected value of `hpc`: '
                    + new_hpc + '. Expected a value in '
                    + "[None, 'cpu', 'gpu'].")

    global __hpc_type
    if __hpc_type != new_hpc:
        exit_handler()
        __hpc_type = new_hpc
        _init_engine()

def get_hpc():
    global __hpc_type

    if __hpc_type == 0:
        return 'cpu'
    if __hpc_type == 1:
        return 'gpu'

    return None

def exit_handler():
    global __engine_initiated
    if __engine_initiated:
        neblina.stop_engine()
        __engine_initiated = False

atexit.register(exit_handler)

def _init_engine():
    r"""
    Initiates neblina-core engine.

    Initiates the engine if it was not previously initiated
    """
    global __engine_initiated
    global __hpc_type
    if not __engine_initiated and __hpc_type is not None:
        # TODO: if not 'neblina' in sys.modules raise ModuleNotFoundError
        neblina_imported = True
        try:
            neblina.init_engine(__hpc_type, 0)
        except NameError:
            neblina_imported = False
        if not neblina_imported:
            raise ModuleNotFoundError(
                "Module neblina was not imported. "
                + "Do you have neblina-core and pyneblina installed?"
            )
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

    # TODO: check if complex automatically?
    # # the complex type from Python is not the same as numpy complex128
    # is_complex = str(v.dtype) == "complex128"
    # more pythonic way of checking if type is complex
    is_complex = np.issubdtype(v.dtype, np.complexfloating)


    n = v.shape[0]

    vec = neblina.load_numpy_array(v)

    # suppose that the vector is going to be used
    # immediately after being transferred
    # TODO: check if this is the case
    neblina.move_vector_device(vec)
    return vec

def retrieve_vector(nbl_vec):
    r"""
    Retrieves vector from the device and converts it to python array.
    By default, it is supposed that the vector is not going to be used in
    other pyneblina calculations,
    thus it is going to be deleted to free memory.
    TODO: get vector dimension(vdim) automatically
    """

    # if a vector is being retrieved.
    # the engine should have been already initiated
    neblina.move_vector_host(nbl_vec)
    py_vec = neblina.retrieve_numpy_array(nbl_vec)

    # if not pynbl_vec.is_complex:
    #     raise NotImplementedError("Cannot retrieve real-only vectors.")
    # py_vec = np.array(
    #             [neblina.vector_get(nbl_vec, 2*i)
    #              + 1j*neblina.vector_get(nbl_vec, 2*i + 1)
    #              for i in range(pynbl_vec.shape)]
    #         )

    # TODO: check if vector is being deleted (or not)

    return py_vec
        
def _send_sparse_matrix(M):
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
    
    # TODO: check if complex automatically?
    is_complex = np.issubdtype(M.dtype, np.complexfloating)
    n = M.shape[0]

    # creates neblina sparse matrix structure
    # TODO: needs better support from pyneblina to
    #   use next instruction (commented).
    #   For example: neblina.sparse_matrix_set works, but in the real case,
    #   it should not be needed to pass the imaginary part as argument.
    #   In addition, there should be a way to
    #   return the matrix and automatically
    #   convert to a matrix of float or of complex numbers accordingly.
    smat = (neblina.sparse_matrix_new(n, n, neblina.COMPLEX) if is_complex
            else neblina.sparse_matrix_new(n, n, neblina.FLOAT))

    for row in range(n):
        start = M.indptr[row]
        end = M.indptr[row + 1]

        # columns must be added in reverse order
        for index in range(end - 1, start - 1, -1):
            col = M.indices[index]

            if is_complex:
                neblina.sparse_matrix_set(smat, row, col,
                                          M[row, col].real,
                                          M[row, col].imag)
            else:
                neblina.sparse_matrix_set(smat, row, col,
                                          M[row, col].real, 0)

    neblina.sparse_matrix_pack(smat)
    neblina.move_sparse_matrix_device(smat)

    return smat

def _send_dense_matrix(M):
    mat = neblina.load_numpy_matrix(M)
    neblina.move_matrix_device(mat)
    return mat

def send_matrix(M):
    if scipy.sparse.issparse(M):
        return _send_sparse_matrix(M)

    return _send_dense_matrix(M)

def retrieve_matrix(nbl_mat):

    try:
        neblina.move_matrix_host(nbl_mat)
        mat = neblina.retrieve_numpy_matrix(nbl_mat)
    except:
        raise NotImplementedError(
            "Cannot retrieve sparse matrix."
        )

    return mat

def multiply_matrix_vector(nbl_mat, nbl_vec, is_sparse):
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
    if is_sparse:
        nbl_vec = neblina.sparse_matvec_mul(nbl_vec, nbl_mat)
    else:
        nbl_vec = neblina.matvec_mul(nbl_vec, nbl_mat)

    return nbl_vec

def multiply_matrices(nbl_A, nbl_B):
    return neblina.mat_mul(nbl_A, nbl_B)

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

    pA = send_matrix(A)

    T = np.eye(A.shape[0], dtype=A.dtype)
    pT = send_matrix(T)

    M = np.eye(A.shape[0], dtype=A.dtype)
    pM = send_matrix(M)

    for i in range(1, n + 1):
        pT = neblina.mat_mul(pT, pA)
        pT = neblina.scalar_mat_mul(1/i, pT)
        pM = neblina.mat_add(pM, pT)

    #return neblina.retrieve_numpy_matrix(pM)
    return retrieve_matrix(pM)
