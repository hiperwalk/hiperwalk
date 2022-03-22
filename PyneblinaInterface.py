def vec_f_(V):
    n = V.shape[0]
    vec_f = vector_new(n, complex_)
    for i in range(n):
        vector_set(vec_f, i, V[i].real, V[i].imag)
    return vec_f
        
def smat_f_(M):
    n = M.shape[0]

    smat_f = sparse_matrix_new(n, n, complex_)
    
    row = 0
    next_row_ind = M.indptr[1]
    j = 2
    for i in range(len(M.data)):
        while i == next_row_ind:
            row += 1
            next_row_ind = M.indptr[j]
            j += 1
            
        col = M.indices[i]
        sparse_matrix_set(smat_f, row, col, M[row, col].real, M[row, col].imag)

    return smat_f
