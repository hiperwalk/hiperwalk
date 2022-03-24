from neblina import *

#Declares data types according to neblina-core
#TODO: make it so it is not hardcoded
NEBLINA_FLOAT = 2
NEBLINA_COMPLEX = 13 

def vec_f_(V):
    n = V.shape[0]
    vec_f = vector_new(n, complex_)
    for i in range(n):
        vector_set(vec_f, i, V[i].real, V[i].imag)
    return vec_f
        
# Transfers a sparse Matrix (M) stored in csr format to Neblina-core
# by default, a matrix with complex elements is expected.
# If the matrix has only real elements, invoke it by using
# TransferSparseMatrix(M, False); this saves half the memory that would be used.
#TODO: Add tests
#   - Transfer and check real Matrix
#   - Transfer and check complex Matrix
#TODO: isn't there a way for neblina-core to use the csr matrix directly?
#   In order to avoid double memory usage
def TransferSparseMatrix(M, isComplex=True):
    n = M.shape[0]

    #creates neblina sparse matrix structure
    smat = sparse_matrix_new(n, n, NEBLINA_COMPLEX) if isComplex else
        sparse_matrix_new(n, n, NEBLINA_FLOAT)
    
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
            sparse_matrix_set(smat, row, col, M[row, col])

    return smat
