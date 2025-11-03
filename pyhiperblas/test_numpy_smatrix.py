import numpy as np
from scipy.sparse import random

import neblina as nbl

n = 100
# Set the seed for reproducibility (optional)
np.random.seed(42)

# Create sparse matrices with float numbers
matrix1 = random(n, n, density=0.2, format='csr')
matrix2 = random(n, n, density=0.2, format='csr')

# Perform matrix multiplication
result = matrix1.dot(matrix2)

# Create sparse matrices with complex numbers
matrix1_complex = random(n, n, density=0.2, format='lil', data_rvs=np.random.random, dtype=np.complex128)
matrix2_complex = random(n, n, density=0.2, format='lil', data_rvs=np.random.random, dtype=np.complex128)

# Perform matrix multiplication
result_complex = matrix1_complex.dot(matrix2_complex)

###########################################################
# neblina calculations
###########################################################

nbl.init_engine(nbl.CPU,0)
smat1 = nbl.sparse_matrix_new(n, n, nbl.FLOAT)
smat2 = nbl.sparse_matrix_new(n, n, nbl.FLOAT)

for i in range(0,n):
    for j in range(0,n):
        nbl.sparse_matrix_set(smat1, i, j, matrix1[i,j], 0)
        nbl.sparse_matrix_set(smat2, i, j, matrix2[i,j], 0)

nbl.sparse_matrix_pack(smat1)
nbl.move_sparse_matrix_device(smat1)

nbl.sparse_matrix_pack(smat2)
nbl.move_sparse_matrix_device(smat2)

res = nbl.mat_mul(smat1, smat2)
print(2)
nbl.move_matrix_host(res)

for i in range(0,n):
    for j in range(0,n):
        assert nbl.matrix_get(res,i,j) == result[i,j]


smat_c1 = nbl.sparse_matrix_new(n, n, nbl.COMPLEX)
smat_c2 = nbl.sparse_matrix_new(n, n, nbl.COMPLEX)

for i in range(0,n):
    for j in range(0,n):
        nbl.sparse_matrix_set(smat_c1, i, j, matrix1_complex[i,j].real, matrix1_complex[i,j].imag)
        nbl.sparse_matrix_set(smat_c2, i, j, matrix2_complex[i,j].real, matrix2_complex[i,j].imag)

nbl.sparse_matrix_pack(smat_c1)
nbl.move_sparse_matrix_device(smat_c1)

nbl.sparse_matrix_pack(smat_c2)
nbl.move_sparse_matrix_device(smat_c2)

res = nbl.mat_mul(smat_c1, smat_c2)

nbl.move_matrix_host(res)

for i in range(0,n):
    for j in range(0,n):
        assert nbl.matrix_get(res,2*i,2*j) == result_complex[i,j].real
        assert nbl.matrix_get(res,2*i,2*j + 1) == result_complex[i,j].imag
        assert nbl.matrix_get(res, 2*i, 2*j) + nbl.matrix_get(res, 2*i, 2*j + 1)*1j == result_complex[i,j]



nbl.stop_engine()

# Create another matrix for comparison
# comparison_matrix = np.random.random((n, n))

# Compare the result with the comparison matrix item by item
# comparison_result = np.allclose(result, comparison_matrix)

# Print the comparison result
# if comparison_result:
#     print("The calculated result is the same as the comparison matrix.")
# else:
#     print("The calculated result is different from the comparison matrix.")
