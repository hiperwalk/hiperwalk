import neblina
from numpy import array as np_array
from constants import NEBLINA_FLOAT, NEBLINA_COMPLEX

# Transfers a vector (v) to Neblina-core, and moves it
# to the device to be used.
# Returns a pointer to this vector
# (needed to call other pyneblina functions).
# by default, a vector with complex entries is expected.
# If the matrix has only real entries, invoke this function by
# TransferVector(v, False); this saves half the memory that would be used.
# TODO: is there a way to move the vector to the device directly?
# I think an auxiliary vector is beign created,
# thus twice the memory needed is being used
def send_vector(v, is_complex=True):
    # TODO: check if is_complex automatically?
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
    return vec

# Retrieves vector from the device and converts it to python array.
# By default, it is supposed that the vector is not going to be used in
# other pyneblina calculations,
# thus it is going to be deleted to free memory.
# TODO: get vector dimension(vdim) automatically
def retrieve_vector(v, vdim, delete_vector=True):
    nbl_vec = neblina.move_vector_host(v)

    # TODO: check type automatically, for now suppose it is only complex
    py_vec = np_array(
                [neblina.vector_get(nbl_vec, 2*i)
                 + 1j*neblina.vector_get(nbl_vec, 2*i + 1)
                 for i in range(vdim)]
            )

    # TODO: check if vector is being deleted (or not)
    # according to the demand
    if delete_vector:
        print('TODO: vector_delete from pyneblina is not available')
        # vector_delete(v)
    print('TODO: vector_delete from pyneblina is not available')
    # vector_delete(nbl_vec)

    return py_vec
        
# Transfers a sparse Matrix (M) stored in csr format to Neblina-core and
# moves it to the device (ready to be used).
# By default, a matrix with complex elements is expected.
# If the matrix has only real elements, invoke this function by
# TransferSparseMatrix(M, False);
# this saves half the memory that would be used.
# TODO: Add tests
#   - Transfer and check real Matrix
#   - Transfer and check complex Matrix
# TODO: isn't there a way for neblina-core to use the csr matrix directly?
#   In order to avoid double memory usage
def send_sparse_matrix(M, is_complex=True):
    # TODO: check if is_complex automatically?
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

    return smat

def multiply_sparse_matrix_vector(smat, vec, is_complex=True):
    """
    Request matrix multiplication to neblina.

    Multiplies the maatrix by the vector, i.e. ``smat @ vec``.

    Parameters
    ----------
    smat
        neblina sparse matrix object
    vec
        neblina vector object
    is_complex : bool default=True
        Whether or not smat or vec is complex.
        .. todo::
            Not implemented.
            neblina implementation for the real case is required.

    Returns
    -------
    Neblina vector object resulted from matrix multiplication.

    See Also
    --------
    send_sparse_matrix : returns a neblina sparse matrix object
    send_vector : returns a neblina vector object
    """

    # TODO: check if is_complex automatically?
    # TODO: ask to change parameter order
    return neblina.sparse_matvec_mul(vec, smat)
