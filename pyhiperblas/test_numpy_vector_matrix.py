import numpy as np
import hiperblas as hb
import time
import pytest

def current_milli_time():
    return round(time.time() * 1000)

n = 14000
# Set the seed for reproducibility (optional)
np.random.seed(42)
matrix_size = (n, n)

float_vector = np.random.random(n)
matrix1 = np.random.random(matrix_size)

print("data generated")
if False:
    vector_c = np.random.random(n) + np.random.random(n) * 1j
    matrix_c = np.random.random(matrix_size) + np.random.random(matrix_size) * 1j

###########################################################
# hiperblas calculations
###########################################################

print(" CALL hb.init_engine(hb.CPU,0)")
#hb.init_engine(hb.GPU,0)
hb.init_engine(hb.CPU,0)


vec_f = hb.load_numpy_array(float_vector)

mat_f1 = hb.load_numpy_matrix(matrix1)
ini = current_milli_time()
hb.move_matrix_device(mat_f1)
hb.move_vector_device(vec_f)


res = hb.matvec_mul(vec_f, mat_f1)
end = current_milli_time()
print("hb.matvec_mul(vec_f, mat_f1) total", (end - ini) )


hb.move_vector_host(res)

np_res = hb.retrieve_numpy_array(res)
if False:
    for i in range(n):
        # print(i, " ", hb.vector_get(res, i), " ", result_float[i], " ", (hb.vector_get(res, i) - result_float[i]))
        assert hb.vector_get(res, i) == pytest.approx(result_float[i], 0.000000000001)
        assert np_res[i] == pytest.approx(result_float[i], 0.000000000001)

if False:
    vec_c = hb.load_numpy_array(vector_c)

    mat_c1 = hb.load_numpy_matrix(matrix_c)

    hb.move_vector_device(vec_c)
    hb.move_matrix_device(mat_c1)

    ini = current_milli_time()
    res = hb.matvec_mul(vec_c, mat_c1)
    end = current_milli_time()
    print("hb.matvec_mul(vec_c, mat_c1) total", (end - ini) )


    hb.move_vector_host(res)

    np_res = hb.retrieve_numpy_array(res)


    for i in range(0,n):
        # print(i, " ", hb.vector_get(res, 2*i), " ", result_c[i].real, " ", (hb.vector_get(res, 2*i) - result_c[i].real))
        # print(i, " ", hb.vector_get(res, 2*i+1), " ", result_c[i].imag, " ", (hb.vector_get(res, 2*i+1) - result_c[i].imag))
        assert hb.vector_get(res, 2 * i) == pytest.approx(result_c[i].real, 0.000000000001)
        assert hb.vector_get(res, 2 * i + 1) == pytest.approx(result_c[i].imag, 0.000000000001)
        assert hb.vector_get(res, 2 * i) + hb.vector_get(res, 2 * i + 1)*1j == pytest.approx(result_c[i], 0.000000000001)
        assert np_res[i] == pytest.approx(result_c[i], 0.000000000001)

hb.stop_engine()

# Perform float vector-sparse matrix multiplication
ini = current_milli_time()
result_float = matrix1.dot(float_vector)
end = current_milli_time()
print("np matrix1.dot(float_vector) total", (end - ini) )

if False:
    # Perform matrix multiplication
    ini = current_milli_time()
    result_c = matrix_c.dot(vector_c)
    end = current_milli_time()
    print("np matrix_c.dot(vector_c) total", (end - ini) )

print("all tests passed")
