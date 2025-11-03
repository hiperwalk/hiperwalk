import numpy as np
from scipy.sparse import random
import pytest
    
import neblina as nbl
    
def main():

    n = 20
    n = 3
    reps = 1 # 10
    # Set the seed for reproducibility (optional)
    np.random.seed(42)
    float_vector = np.random.random(n)
    float_vector = np.ones(n)
    #float_matrix = random(n, n, density=1.0, format='lil', data_rvs=np.random.random)
    #float_matrix = random(n, n, density=1.0, format='lil') #, data_rvs=np.random.random)

    float_matrix = np.fromfunction(lambda p, q: 1.0 + 1.0 * (p + 10*q), (n, n), dtype=float)

    print("float_matrix=\n", float_matrix)
    
    ###########################################################
    # hiperblas calculations
    ###########################################################
    nbl.init_engine(nbl.CPU,0)
    vecF = nbl.vector_new(n, nbl.FLOAT)
    for i in range(n):
        nbl.vector_set(vecF, i, float_vector[i], 0)
    
    smatF = nbl.sparse_matrix_new(n, n, nbl.FLOAT)
    for i in range(0,n):
        for j in range(0,n):
            nbl.sparse_matrix_set(smatF, i, j, float_matrix[i,j], 0)

    nbl.sparse_matrix_pack(smatF)
    nbl.sparse_matrix_print(smatF);

    #nbl.move_sparse_matrix_device(smatF)
    #nbl.move_vector_device(vecF)

    res = nbl.sparse_matvec_mul(vecF, smatF)
    nbl.move_vector_host(res)

    for i in range(0,n):
        print("nbl.vector_get(res,i)", nbl.vector_get(res,i))
    return


    print("tests complete")
    nbl.stop_engine()

    return

    vec_f = nbl.vector_new(n, nbl.FLOAT)
    nbl.move_sparse_matrix_device(smat1)
    nbl.move_vector_device(vec_f)

    res = nbl.sparse_matvec_mul(vec_f, smat1)
    print(" return, em main()\n"); return
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

main()
