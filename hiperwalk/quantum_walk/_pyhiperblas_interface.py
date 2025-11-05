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
__hpc_type = "CPU"

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

def retrieve_vector(nbl_vec):
    r"""
    Retrieves vector from the device and converts it to python array.
    By default, it is supposed that the vector is not going to be used in
    other pyhiperblas calculations,
    thus it is going to be deleted to free memory.
    TODO: get vector dimension(vdim) automatically
    """

    # if a vector is being retrieved.
    # the engine should have been already initiated
    hiperblas.move_vector_host(nbl_vec)
    py_vec = hiperblas.retrieve_numpy_array(nbl_vec)

    # if not pynbl_vec.is_complex:
    #     raise NotImplementedError("Cannot retrieve real-only vectors.")
    # py_vec = np.array(
    #             [hiperblas.vector_get(nbl_vec, 2*i)
    #              + 1j*hiperblas.vector_get(nbl_vec, 2*i + 1)
    #              for i in range(pynbl_vec.shape)]
    #         )

    # TODO: check if vector is being deleted (or not)

    return py_vec
        
def _send_sparse_matrix(M):
    r"""
    Transfers a sparse Matrix (M) stored in csr format to Hiperblas-core and
    moves it to the device (ready to be used).
    By default, a matrix with complex elements is expected.
    If the matrix has only real elements, invoke this function by
    TransferSparseMatrix(M, False);
    this saves half the memory that would be used.
    TODO: Add tests
      - Transfer and check real Matrix
      - Transfer and check complex Matrix
    TODO: isn't there a way for hiperblas-core to use the csr matrix directly?
      In order to avoid double memory usage
    """
    
    print("BD, em hiperwalk/quantum_walk/_pyhiperblas_interface.py: def _send_sparse_matrix(M)");
    # TODO: check if complex automatically?
    print("M.dtype=",M.dtype, ", np.complexfloating=", np.complexfloating)
    is_complex = np.issubdtype(M.dtype, np.complexfloating)
    n = M.shape[0]

    # creates hiperblas sparse matrix structure
    # TODO: needs better support from pyhiperblas to
    #   use next instruction (commented).
    #   For example: hiperblas.sparse_matrix_set works, but in the real case,
    #   it should not be needed to pass the imaginary part as argument.
    #   In addition, there should be a way to
    #   return the matrix and automatically
    #   convert to a matrix of float or of complex numbers accordingly.
    print(f"BD, is_complex={is_complex}")
    print(f"BD, hiperblas.COMPLEX={hiperblas.COMPLEX}")
    print(f"BD, hiperblas.FLOAT={hiperblas.FLOAT}"); #exit()
    smat = (hiperblas.sparse_matrix_new(n, n, hiperblas.COMPLEX) if is_complex
            else hiperblas.sparse_matrix_new(n, n, hiperblas.FLOAT))

    # vBD = hiperblas.vector_new(n, hiperblas.FLOAT); hiperblas.print_vectorT(vBD)

    hiperblas.smatrixConnect(smat, M ); 
    #hiperblas.sparse_matrix_pack(smat)
    #if n < 30 :
    #    print("BD3, em hiperwalk/quantum_walk/_pyhiperblas_interface.py: def _send_sparse_matrix(M), CALL hiperblas.sparse_matrix_print(smat)");
    #    hiperblas.sparse_matrix_print(smat)
    hiperblas.move_sparse_matrix_device(smat)
    return smat

    #KOR print("em def _send_sparse_matrix(M); A "); 
    #KOR for row in range(n):
    #KOR     start = M.indptr[row]; end   = M.indptr[row + 1]
    #KOR     # columns must be added in reverse order
    #KOR     for index in range(end - 1, start - 1, -1):
    #KOR         col = M.indices[index]
    #KOR         if is_complex:
    #KOR             hiperblas.sparse_matrix_set(smat, row, col, M[row, col].real, M[row, col].imag)
    #KOR         else:
    #KOR             hiperblas.sparse_matrix_set(smat, row, col, M[row, col].real, 0)

    #KOR print("em def _send_sparse_matrix(M);") # quit()"); quit()
    #KOR hiperblas.sparse_matrix_pack(smat)
    #KOR hiperblas.move_sparse_matrix_device(smat)

    print("em hiperwalk/quantum_walk/_pyhiperblas_interface.py: def _send_sparse_matrix(M), FIM"); # exit()
    return smat

def _send_dense_matrix(M):
    print("em hiperwalk/quantum_walk/_pyhiperblas_interface.py: def _send_dense_matrix(M)")
    mat = hiperblas.load_numpy_matrix(M)
    hiperblas.move_matrix_device(mat)
    return mat

def send_matrix(M):
    print("BD, em hiperwalk/quantum_walk/_pyhiperblas_interface.py: def send_matrix(M)")
    print("BD, M.dtype=",M.dtype, ", np.complexfloating=", np.complexfloating)
    if scipy.sparse.issparse(M):
        s_matrix = _send_sparse_matrix(M)
        #hiperblas.print_smatrix(s_matrix); exit()
        #print("BD, em send_matrix(M): call hiperblas.sparse_matrix_print(s_matrix);")
        #hiperblas.sparse_matrix_print(s_matrix); exit()
        #print("BD, em quantum_walk/_pyhiperblas_interface.py:  send_matrix(M): return sparse_matrix;")
        return s_matrix

    return _send_dense_matrix(M)

def retrieve_matrix(nbl_mat):

    try:
        hiperblas.move_matrix_host(nbl_mat)
        mat = hiperblas.retrieve_numpy_matrix(nbl_mat)
    except:
        raise NotImplementedError(
            "Cannot retrieve sparse matrix."
        )

    return mat

def print_vectorT_py_inter(nbl_vec):
    hiperblas.print_vectorT(nbl_vec)

def sparse_matrix_print(nbl_mat):
    hiperblas.sparse_matrix_print(nbl_mat)

def permute_sparse_matrix(nbl_smatS, nbl_smatC, nbl_smatU):
        print("BD1, em _pyHiperblas_interface.py: def permute_sparse_matrix(nbl_smatS, nbl_smatC, nbl_smatU):")
        #print("BD2, def permute_sparse_matrix, CALL hiperblas.permute_sparse_matrix(")
        hiperblas.permute_sparse_matrix(nbl_smatS, nbl_smatC, nbl_smatU)
        #print("BD3, :def permute_sparse_matrix, exit() = "); exit()
        
        #hiperblas.sparse_matrix_print(nbl_smatU);
        return

def multiply_matrix_vector(nbl_mat, nbl_vecIn, nbl_vecOut, is_sparse):
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
    print("BD, em ./hiperwalk/quantum_walk/_pyhiperblas_interface.py, def multiply_matrix_vector( .., is_sparse=",is_sparse)
    if is_sparse:
        print("BD, em ./hiperwalk/quantum_walk/_pyhiperblas_interface.py, def multiply_matrix_vector, CALL nbl_vec = hiperblas.sparse_matvec_mul, esparsa, para discreto ")
        hiperblas.sparse_matvec_mulBD(nbl_mat, nbl_vecIn, nbl_vecOut)
    else:
        print("BD, em ./hiperwalk/quantum_walk/_pyhiperblas_interface.py, def multiply_matrix_vector, CALL nbl_vec = hiperblas.matvec_mul, DENSA, para continuo ")
        nbl_vecOut = hiperblas.matvec_mul(nbl_vecIn, nbl_mat)

    return 

def multiply_matrices(nbl_A, nbl_B):
    return hiperblas.mat_mul(nbl_A, nbl_B)

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

    #return hiperblas.retrieve_numpy_matrix(pM)
    return retrieve_matrix(pM)

def copy_vector(v):
    res = hiperblas.copy_vector_from_device(v)
    vec = hiperblas.retrieve_numpy_array(res)
    return vec
