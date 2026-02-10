try:
    import hiperblas
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

    print ("BD, em def set_hpc(hpc), hpc=",hpc)
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
        hiperblas.stop_engine()
        __engine_initiated = False

atexit.register(exit_handler)

def _init_engine():
    r"""
    Initiates hiperblas-core engine.

    Initiates the engine if it was not previously initiated
    """
    global __engine_initiated
    global __hpc_type
    if not __engine_initiated and __hpc_type is not None:
        # TODO: if not 'hiperblas' in sys.modules raise ModuleNotFoundError
        hiperblas_imported = True
        try:
            hiperblas.init_engine(__hpc_type, 0)
        except NameError:
            hiperblas_imported = False
        if not hiperblas_imported:
            raise ModuleNotFoundError(
                "Module hiperblas was not imported. "
                + "Do you have hiperblas-core and pyhiperblas installed?"
            )
        __engine_initiated = True

def send_vector(v):
    r"""
    Transfers a vector (v) to Hiperblas-core, and moves it
    to the device to be used.
    Returns a pointer to this vector
    (needed to call other pyhiperblas functions).
    by default, a vector with complex entries is expected.
    If the matrix has only real entries, invoke this function by
    TransferVector(v, False);
    this saves half the memory that would be used.
    TODO: is there a way to move the vector to the device directly?
    I think an auxiliary vector is beign created,
    thus twice the memory needed is being used
    """
    vec = hiperblas.load_numpy_array(v)
    hiperblas.move_vector_device(vec)
    return vec

def retrieve_vector(hpb_vec):
    r"""
    Retrieves vector from the device and converts it to python array.
    By default, it is supposed that the vector is not going to be used in
    other pyhiperblas calculations,
    thus it is going to be deleted to free memory.
    TODO: get vector dimension(vdim) automatically
    """

    # if a vector is being retrieved.
    # the engine should have been already initiated
    hiperblas.move_vector_host(hpb_vec)
    py_vec = hiperblas.retrieve_numpy_array(hpb_vec)
    return py_vec
        
def _send_dense_matrix(M):
    print("em hiperwalk/quantum_walk/_pyhiperblas_interface.py: "
          + "def _send_dense_matrix(M)")
    mat = hiperblas.load_numpy_matrix(M)
    hiperblas.move_matrix_device(mat)
    return mat

def send_matrix(M):
    print("BD, em hiperwalk/quantum_walk/_pyhiperblas_interface.py: "
          + "def send_matrix(M)")
    return _send_dense_matrix(M)

def retrieve_matrix(hpb_mat):

    try:
        hiperblas.move_matrix_host(hpb_mat)
        mat = hiperblas.retrieve_numpy_matrix(hpb_mat)
    except:
        raise NotImplementedError(
            "Cannot retrieve sparse matrix."
        )

    return mat

def multiply_matrix_vector(hpb_mat, hpb_vecIn, hpb_vecOut, is_sparse):
    """
    Request matrix multiplication to hiperblas.

    Multiplies the matrix by the vector, i.e. ``smat @ vec``.

    Parameters
    ----------
    mat : :class:`PyHiperblasMatrix`
        hiperblas matrix object
    vec : :class:`PyHiperblasVector`
        hiperblas vector object

    Returns
    -------
    Hiperblas vector object resulted from matrix multiplication.

    See Also
    --------
    send_sparse_matrix : returns a hiperblas sparse matrix object
    send_vector : returns a hiperblas vector object
    """
    # if a matrix-vector operation is being requested,
    # the engine should have been already initiated
    if is_sparse:
        print("BD, em "
              + "./hiperwalk/quantum_walk/_pyhiperblas_interface.py, "
              + "def multiply_matrix_vector, "
              + "CALL hpb_vec = hiperblas.sparse_matvec_mul, "
              + "esparsa, para discreto")
        hiperblas.sparse_matvec_mul(hpb_mat, hpb_vecIn, hpb_vecOut)
    else:
        print("BD, em "
              + "./hiperwalk/quantum_walk/_pyhiperblas_interface.py, "
              + "def multiply_matrix_vector, "
              + "CALL hpb_vec = hiperblas.matvec_mul, "
              + "DENSA, para continuo")
        hpb_vecOut = hiperblas.matvec_mul(hpb_vecIn, hpb_mat)

    return 

def multiply_matrices(hpb_A, hpb_B):
    return hiperblas.mat_mul(hpb_A, hpb_B)

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
        pT = hiperblas.mat_mul(pT, pA)
        pT = hiperblas.scalar_mat_mul(1/i, pT)
        pM = hiperblas.mat_add(pM, pT)

    return retrieve_matrix(pM)
