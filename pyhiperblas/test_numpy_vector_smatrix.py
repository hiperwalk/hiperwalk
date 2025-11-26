import numpy as np
from scipy.sparse import random
from scipy.sparse import csr_matrix
#import pytest
    
import ctypes

import hiperblas as hb

def main():

    print("em def main():")
    n = 3 # 20
    reps = n - 1 # 10
    float_vector = np.ones(n)
    float_vector[0]=2.1; float_vector[1]=1.1; float_vector[2]=0.1; 

    float_matrix = np.fromfunction(lambda p, q: 1.0 + 1.0 * (10*p + q), (n, n), dtype=float)
    float_matrix[0][2]=0.0; float_matrix[2][0]=0.0

#    print("float_matrix=\n", float_matrix)
    
    ###########################################################
    # hiperblas calculations
    ###########################################################
    hb.init_engine(hb.CPU,0)

    vecF = hb.load_numpy_array(float_vector)
    #vecF = hb.vector_new(n, hb.FLOAT)
    #for i in range(n): hb.vector_set(vecF, i, float_vector[i], 0)
    for i in range(0,n): print(f"vecF[{i}] = {hb.vector_get(vecF,i)}")

    print("Matriz densa:"); print(float_matrix)
    # cria matriz esparsa CSR
    csrMat = csr_matrix(float_matrix);
    print("\nMatriz em formato CSR:"); print(csrMat);
    print("\nArrays internos do CSR:")
    print("indptr   =", csrMat.indptr)    # ponteiros para início de cada linha
    print("indices  =", csrMat.indices)   # índices das colunas
    print("data     =", csrMat.data)      # valores não nulos

    #row_ptr: 0 2 5 7
    #col_idx: 1 0 2 1 0 2 1
    #values: 2.0000 1.0000 13.0000 12.0000 11.0000 23.0000 22.0000

    smatF = hb.sparse_matrix_new(n, n, hb.FLOAT)
    hb.sparse_matrix_print(smatF);
    hb.smatrixConnect(smatF, csrMat);
    hb.sparse_matrix_print(smatF);

    #hb.move_sparse_matrix_device(smatF)
    hb.move_vector_device(vecF)

    res = hb.load_numpy_array(np.zeros(n))
    hb.print_vectorT(res)

    float_vectorRef=float_matrix@float_vector
    hb.sparse_matvec_mul(smatF, vecF, res) # operacao feita no device
    hb.print_vectorT(res)
    print(float_vectorRef)

    float_vector=float_matrix@float_vectorRef
    vecF, res = res, vecF
    hb.sparse_matvec_mul(smatF, vecF, res) # operacao feita no device
    hb.print_vectorT(res)
    print(float_vector)


    return
    exit()

print(" main()")
main()
