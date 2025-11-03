import numpy as np
from scipy.sparse import random
import pytest

import neblina as nbl

n = 20
reps = 3
# Set the seed for reproducibility (optional)
np.random.seed(42)

float_vector = np.random.random(n)
float_matrix = random(n, n, density=0.2, format='lil', data_rvs=np.random.random)

# Perform float vector-sparse matrix multiplication
result_float = float_matrix.tocsr().dot(float_vector)
for i in range(reps):
    result_float = float_matrix.tocsr().dot(result_float)

complex_vector = np.random.random(n) + np.random.random(n) * 1j
complex_matrix = random(n, n, density=0.2, format='lil', data_rvs=np.random.random, dtype=np.complex128)

# Perform complex vector-sparse matrix multiplication
result_complex = complex_matrix.tocsr().dot(complex_vector)
for i in range(reps):
    result_complex = complex_matrix.tocsr().dot(result_complex)

###########################################################
# neblina calculations
###########################################################

nbl.init_engine(nbl.CPU,0)

vec_f = nbl.vector_new(n, nbl.FLOAT)
for i in range(n):
    nbl.vector_set(vec_f, i, float_vector[i], 0)

smat1 = nbl.sparse_matrix_new(n, n, nbl.FLOAT)

for i in range(0,n):
    for j in range(0,n):
        nbl.sparse_matrix_set(smat1, i, j, float_matrix[i,j], 0)

nbl.sparse_matrix_pack(smat1)
nbl.move_sparse_matrix_device(smat1)

nbl.move_vector_device(vec_f)

res = nbl.sparse_matvec_mul(vec_f, smat1)
for i in range(reps):
    res = nbl.sparse_matvec_mul(res, smat1)

nbl.move_vector_host(res)

result_float_array = np.array(result_float)
for i in range(0,n):
    print("nbl.vector_get(res,i)", nbl.vector_get(res,i))
    print("result_float[i]", result_float_array[i])
    print(nbl.vector_get(res,i) - result_float_array[i])
    # assert nbl.vector_get(res,i) == pytest.approx(result_float_array[i], 0.0000000000000001)

vec_c = nbl.vector_new(n, nbl.COMPLEX)

for i in range(n):
    nbl.vector_set(vec_c, i, complex_vector[i].real, complex_vector[i].imag)
smat_c1 = nbl.sparse_matrix_new(n, n, nbl.COMPLEX)

for i in range(0,n):
    for j in range(0,n):
        nbl.sparse_matrix_set(smat_c1, i, j, complex_matrix[i,j].real, complex_matrix[i,j].imag)

nbl.sparse_matrix_pack(smat_c1)
nbl.move_sparse_matrix_device(smat_c1)

nbl.move_vector_device(vec_c)

res = nbl.sparse_matvec_mul(vec_c, smat_c1)

for i in range(reps):
    res = nbl.sparse_matvec_mul(res, smat_c1)

nbl.move_vector_host(res)

for i in range(0,n):
    print("nbl.vector_get(res, 2 * i)=", nbl.vector_get(res, 2 * i), " result_complex[i].real=",result_complex[i].real)
    print("nbl.vector_get(res, 2 * i +1)=", nbl.vector_get(res, 2 * i + 1), " result_complex[i].imag=",result_complex[i].imag)
    print("diff real=", (nbl.vector_get(res, 2 * i) - result_complex[i].real))
    print("diff imag =", (nbl.vector_get(res, 2 * i + 1) - result_complex[i].imag))
    # assert nbl.matrix_get(res,2*i,2*j) == result_complex[i,j].real
    # assert nbl.matrix_get(res,2*i,2*j + 1) == result_complex[i,j].imag
    # assert nbl.matrix_get(res, 2*i, 2*j) + nbl.matrix_get(res, 2*i, 2*j + 1)*1j == result_complex[i,j]
    assert nbl.vector_get(res, 2 * i) == pytest.approx(result_complex[i].real, 0.000000000000001)
    assert nbl.vector_get(res, 2 * i + 1) == pytest.approx(result_complex[i].imag, 0.000000000000001)

print("tests complete")
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
