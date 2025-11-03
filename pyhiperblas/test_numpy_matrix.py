import numpy as np
import neblina as nbl
import time
import pytest

def current_milli_time():
    return round(time.time() * 1000)

# n = 26244
n = 1521
# Set the seed for reproducibility (optional)
np.random.seed(42)

# Create the first matrix with random data
matrix1 = np.random.random((n, n))

# Create the second matrix with random data
matrix2 = np.random.random((n, n))

ini = current_milli_time()
# Perform matrix multiplication
result = np.dot(matrix1, matrix2)
end = current_milli_time()
print("np.dot(matrix1, matrix2) float total", (end - ini) )

matrixc1 = np.random.random((n, n)) + np.random.random((n, n)) * 1j

# Create the second matrix with random complex data
matrixc2 = np.random.random((n, n)) + np.random.random((n, n)) * 1j

ini = current_milli_time()
resultc = np.dot(matrixc1, matrixc2)
end = current_milli_time()
print("np.dot(matrixc1, matrixc2) complex total", (end - ini) )

###########################################################
# neblina calculations
###########################################################

nbl.init_engine(nbl.GPU,0)
# mat_c1 = nbl.matrix_new(n, n, nbl.FLOAT)
# mat_c2 = nbl.matrix_new(n, n, nbl.FLOAT)

ini = current_milli_time()
mat_f1 = nbl.load_numpy_matrix(matrix1)
mat_f2 = nbl.load_numpy_matrix(matrix2)
end = current_milli_time()
print("nbl.load_numpy_matrix total", (end - ini) )

ini = current_milli_time()
nbl.move_matrix_device(mat_f1)
nbl.move_matrix_device(mat_f2)


res = nbl.mat_mul(mat_f1, mat_f2)
end = current_milli_time()
print("nbl.mat_mul(mat_f1, mat_f2) float total", (end - ini) )

nbl.move_matrix_host(res)

np_res = nbl.retrieve_numpy_matrix(res)
# print(np_res.shape)
# print(result.shape)

if True:
    for i in range(0,n):
        for j in range(0,n):
            # print( nbl.matrix_get(res,i,j), " ",  result[i,j])
            assert nbl.matrix_get(res,i,j) == pytest.approx(result[i,j], 0.0000000000000001)
            assert np_res[i,j] == pytest.approx(result[i,j], 0.0000000000000001)

# mat_c1 = nbl.matrix_new(n, n, nbl.COMPLEX)
# mat_c2 = nbl.matrix_new(n, n, nbl.COMPLEX)

ini = current_milli_time()
mat_c1 = nbl.load_numpy_matrix(matrixc1)
mat_c2 = nbl.load_numpy_matrix(matrixc2)
end = current_milli_time()
print("nbl.load_numpy_matrix complex total", (end - ini) )

nbl.move_matrix_device(mat_c1)
nbl.move_matrix_device(mat_c2)

ini = current_milli_time()
res = nbl.mat_mul(mat_c1, mat_c2)
end = current_milli_time()
print("nbl.mat_mul(mat_c1, mat_c2) complex total", (end - ini) )

nbl.move_matrix_host(res)

np_res = nbl.retrieve_numpy_matrix(res)
print(np_res.shape)
print(resultc.shape)

if True:
    for i in range(0,n):
        for j in range(0,n):
            assert nbl.matrix_get(res,2*i,2*j) == pytest.approx(resultc[i,j].real, 0.0000000000000001)
            assert nbl.matrix_get(res,2*i,2*j + 1) == pytest.approx(resultc[i,j].imag, 0.0000000000000001)
            # print( nbl.matrix_get(res, 2*i, 2*j) + nbl.matrix_get(res, 2*i, 2*j + 1)*1j ," ", resultc[i,j])
            # print( nbl.matrix_get(res, 2*i, 2*j), " ", nbl.matrix_get(res, 2*i, 2*j + 1)*1j ," ", resultc[i,j])
            assert nbl.matrix_get(res, 2*i, 2*j) + nbl.matrix_get(res, 2*i, 2*j + 1)*1j == pytest.approx(resultc[i,j], 0.0000000000000001)
            assert np_res[i,j] == pytest.approx(resultc[i,j], 0.0000000000000001)



nbl.stop_engine()
print("all tests passed")