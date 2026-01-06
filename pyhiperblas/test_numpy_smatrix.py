import numpy as np
from scipy.sparse import random

import hiperblas as hb

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
# hiperblas calculations
###########################################################

hb.init_engine(hb.CPU,0)
smat1 = hb.sparse_matrix_new(n, n, hb.FLOAT)
smat2 = hb.sparse_matrix_new(n, n, hb.FLOAT)

for i in range(0,n):
    for j in range(0,n):
        hb.sparse_matrix_set(smat1, i, j, matrix1[i,j], 0)
        hb.sparse_matrix_set(smat2, i, j, matrix2[i,j], 0)

hb.sparse_matrix_pack(smat1)
hb.move_sparse_matrix_device(smat1)

hb.sparse_matrix_pack(smat2)
hb.move_sparse_matrix_device(smat2)

res = hb.mat_mul(smat1, smat2)
print(2)
hb.move_matrix_host(res)

for i in range(0,n):
    for j in range(0,n):
        assert hb.matrix_get(res,i,j) == result[i,j]


smat_c1 = hb.sparse_matrix_new(n, n, hb.COMPLEX)
smat_c2 = hb.sparse_matrix_new(n, n, hb.COMPLEX)

for i in range(0,n):
    for j in range(0,n):
        hb.sparse_matrix_set(smat_c1, i, j, matrix1_complex[i,j].real, matrix1_complex[i,j].imag)
        hb.sparse_matrix_set(smat_c2, i, j, matrix2_complex[i,j].real, matrix2_complex[i,j].imag)

hb.sparse_matrix_pack(smat_c1)
hb.move_sparse_matrix_device(smat_c1)

hb.sparse_matrix_pack(smat_c2)
hb.move_sparse_matrix_device(smat_c2)

res = hb.mat_mul(smat_c1, smat_c2)

hb.move_matrix_host(res)

for i in range(0,n):
    for j in range(0,n):
        assert hb.matrix_get(res,2*i,2*j) == result_complex[i,j].real
        assert hb.matrix_get(res,2*i,2*j + 1) == result_complex[i,j].imag
        assert hb.matrix_get(res, 2*i, 2*j) + hb.matrix_get(res, 2*i, 2*j + 1)*1j == result_complex[i,j]



hb.stop_engine()

# Create another matrix for comparison
# comparison_matrix = np.random.random((n, n))

# Compare the result with the comparison matrix item by item
# comparison_result = np.allclose(result, comparison_matrix)

# Print the comparison result
# if comparison_result:
#     print("The calculated result is the same as the comparison matrix.")
# else:
#     print("The calculated result is different from the comparison matrix.")
