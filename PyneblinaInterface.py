from neblina import *
from numpy import array as np_array

#Declares data types according to neblina-core
#TODO: make it so it is not hardcoded
NEBLINA_FLOAT = 2
NEBLINA_COMPLEX = 13 

# Transfers a vector (v) to Neblina-core, and moves it to the device to be used.
# Returns a pointer to this vector (needed to call other pyneblina functions).
# by default, a vector with complex entries is expected.
# If the matrix has only real entries, invoke this function by
# TransferVector(v, False); this saves half the memory that would be used.
# TODO: is there a way to move the vector to the device directly?
#   I think an auxiliary vector is beign created, thus twice the memory needed is being used
def NeblinaSendVector(v, isComplex=True):
    #TODO: check if isComplex automatically?
    n = v.shape[0]
    #TODO: needs better support from pyneblina to use next instruction (commented).
    #   For example: vector_set works, but in the real case, it should not be needed to
    #   pass the imaginary part as argument.
    #   In addition, there should be a way to return a vector and automatically
    #   convert to an array of float or of complex numbers accordingly.
    #vec = vector_new(n, NEBLINA_COMPLEX) if isComplex else vector_new(n, NEBLINA_FLOAT)
    vec = vector_new(n, NEBLINA_COMPLEX)

    if isComplex:
        for i in range(n):
            vector_set(vec, i, v[i].real, v[i].imag)
    else:
        for i in range(n):
            #TODO: Pyneblina needs to accept 3 arguments only instead of 4?
            #TODO: check if vector_set is idetifying the vector type right
            #(i.e. real and not complex)
            vector_set(vec, i, v[i], 0)

    #suppose that the vector is going to be used immediately after being transferred
    #TODO: check if this is the case
    move_vector_device(vec)
    return vec

#Retrieves vector from the device and converts it to python array.
#By default, it is supposed that the vector is not going to be used in
#other pyneblina calculations, thus it is going to be deleted to free memory.
#TODO: get vector dimension(vdim) automatically
def NeblinaRetrieveVector(v, vdim, deleteVector=True):
    nbl_vec = move_vector_host(v)

    #TODO: check type automatically, for now suppose it is only complex
    py_vec = np_array(
                [vector_get(nbl_vec, 2*i) + 1j*vector_get(nbl_vec, 2*i+1) for i in range(vdim)]
            )

    #TODO: check if vector is being deleted (or not) according to the demand
    if deleteVector:
        print('TODO: vector_delete from pyneblina is not available')
        #vector_delete(v)
    print('TODO: vector_delete from pyneblina is not available')
    #vector_delete(nbl_vec)

    return py_vec
        
# Transfers a sparse Matrix (M) stored in csr format to Neblina-core and
# moves it to the device (ready to be used).
# By default, a matrix with complex elements is expected.
# If the matrix has only real elements, invoke this function by
# TransferSparseMatrix(M, False); this saves half the memory that would be used.
#TODO: Add tests
#   - Transfer and check real Matrix
#   - Transfer and check complex Matrix
#TODO: isn't there a way for neblina-core to use the csr matrix directly?
#   In order to avoid double memory usage
def NeblinaSendSparseMatrix(M, isComplex=True):
    #TODO: check if isComplex automatically?
    n = M.shape[0]

    #creates neblina sparse matrix structure
    #TODO: needs better support from pyneblina to use next instruction (commented).
    #   For example: sparse_matrix_set works, but in the real case, it should not be needed to
    #   pass the imaginary part as argument.
    #   In addition, there should be a way to return the matrix and automatically
    #   convert to a matrix of float or of complex numbers accordingly.
    #smat = sparse_matrix_new(n, n, NEBLINA_COMPLEX) if isComplex else sparse_matrix_new(
    #        n, n, NEBLINA_FLOAT)
    smat = sparse_matrix_new(n, n, NEBLINA_COMPLEX)
    
    #inserts elements into neblina sparse matrix
    row = 0
    next_row_ind = M.indptr[1]
    j = 2
    for i in range(len(M.data)):
        while i == next_row_ind:
            row += 1
            next_row_ind = M.indptr[j]
            j += 1
            
        col = M.indices[i]
        if isComplex:
            sparse_matrix_set(smat, row, col, M[row, col].real, M[row, col].imag)
        else:
            #TODO: Pynebliena needs to accept 4 arguments instead of 5?
            #TODO: check if smatrix_set_real_value is beign called instead of smatrix_set_complex_value
            sparse_matrix_set(smat, row, col, M[row, col], 0) 

    sparse_matrix_pack(smat) #TODO: is this needed?
    move_sparse_matrix_device(smat)

    return smat
