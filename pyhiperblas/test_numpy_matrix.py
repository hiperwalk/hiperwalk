import numpy as np
import hiperblas as hb
import time
import pytest

def current_milli_time():
    return round(time.time() * 1000)

# n = 26244
n = 1521
n = 5
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

#hb.init_engine(hb.GPU,0)
hb.init_engine(hb.CPU,0)
# mat_c1 = hb.matrix_new(n, n, hb.FLOAT)
# mat_c2 = hb.matrix_new(n, n, hb.FLOAT)

ini = current_milli_time()
mat_f1 = hb.load_numpy_matrix(matrix1)
mat_f2 = hb.load_numpy_matrix(matrix2)
end = current_milli_time()
print("hb.load_numpy_matrix total", (end - ini) )

ini = current_milli_time()
hb.move_matrix_device(mat_f1)
hb.move_matrix_device(mat_f2)


res = hb.mat_mul(mat_f1, mat_f2)
end = current_milli_time()
print("hb.mat_mul(mat_f1, mat_f2) float total", (end - ini) )

hb.move_matrix_host(res)

np_res = hb.retrieve_numpy_matrix(res)
# print(np_res.shape)
# print(result.shape)

if True:
    for i in range(0,n):
        for j in range(0,n):
            print( hb.matrix_get(res,i,j), " ",  result[i,j])
            #assert hb.matrix_get(res,i,j) == pytest.approx(result[i,j], 0.0000000000000001)
            #assert np_res[i,j] == pytest.approx(result[i,j], 0.0000000000000001)

# mat_c1 = hb.matrix_new(n, n, hb.COMPLEX)
# mat_c2 = hb.matrix_new(n, n, hb.COMPLEX)

ini = current_milli_time()
mat_c1 = hb.load_numpy_matrix(matrixc1)
mat_c2 = hb.load_numpy_matrix(matrixc2)
end = current_milli_time()
print("hb.load_numpy_matrix complex total", (end - ini) )

hb.move_matrix_device(mat_c1)
hb.move_matrix_device(mat_c2)

ini = current_milli_time()
res = hb.mat_mul(mat_c1, mat_c2)
end = current_milli_time()
print("hb.mat_mul(mat_c1, mat_c2) complex total", (end - ini) )

hb.move_matrix_host(res)

np_res = hb.retrieve_numpy_matrix(res)
print(np_res.shape)
print(resultc.shape)

if True:
    for i in range(0,n):
        for j in range(0,n):
            assert hb.matrix_get(res,2*i,2*j) == pytest.approx(resultc[i,j].real, 0.0000000000000001)
            assert hb.matrix_get(res,2*i,2*j + 1) == pytest.approx(resultc[i,j].imag, 0.0000000000000001)
            # print( hb.matrix_get(res, 2*i, 2*j) + hb.matrix_get(res, 2*i, 2*j + 1)*1j ," ", resultc[i,j])
            # print( hb.matrix_get(res, 2*i, 2*j), " ", hb.matrix_get(res, 2*i, 2*j + 1)*1j ," ", resultc[i,j])
            assert hb.matrix_get(res, 2*i, 2*j) + hb.matrix_get(res, 2*i, 2*j + 1)*1j == pytest.approx(resultc[i,j], 0.0000000000000001)
            assert np_res[i,j] == pytest.approx(resultc[i,j], 0.0000000000000001)



hb.stop_engine()
print("all tests passed")
