from neblina import *

#Declares data types according to neblina-core
#TODO: make it so it is not hardcoded
NEBLINA_FLOAT = 2
NEBLINA_COMPLEX = 13 

# Transfers a vector (v) to Neblina-core.
# by default, a vector with complex entries is expected.
# If the matrix has only real entries, invoke this function by
# TransferVector(v, False); this saves half the memory that would be used.
def TransferVector(v, isComplex=True):
    n = v.shape[0]
    vec = vector_new(n, NEBLINA_COMPLEX) if isComplex else vector_new(n, NEBLINA_FLOAT)

    if isComplex:
        for i in range(n):
            vector_set(vec, i, v[i].real, v[i].imag)
        return vec

    # else is real
    for i in range(n):
        #TODO: Pyneblina needs to accept 3 arguments only instead of 4?
        #TODO: check if vector_set is idetifying the vector type right (i.e. real and not complex)
        vector_set(vec, i, v[i], 0)
    return vec
        
# Transfers a sparse Matrix (M) stored in csr format to Neblina-core.
# By default, a matrix with complex elements is expected.
# If the matrix has only real elements, invoke this function by
# TransferSparseMatrix(M, False); this saves half the memory that would be used.
#TODO: Add tests
#   - Transfer and check real Matrix
#   - Transfer and check complex Matrix
#TODO: isn't there a way for neblina-core to use the csr matrix directly?
#   In order to avoid double memory usage
def TransferSparseMatrix(M, isComplex=True):
    n = M.shape[0]

    #creates neblina sparse matrix structure
    smat = sparse_matrix_new(n, n, NEBLINA_COMPLEX) if isComplex else sparse_matrix_new(
            n, n, NEBLINA_FLOAT)
    
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

    return smat
