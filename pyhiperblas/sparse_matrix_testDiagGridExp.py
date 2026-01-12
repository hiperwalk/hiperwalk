import numpy as np
import pytest
from scipy.sparse import random
from scipy.sparse import block_diag, csr_matrix
import hiperblas as hb

def vector_normL2_one (n_):
   v = np.random.randn(n_)    # vetor aleatório
   v = v / np.linalg.norm(v) # normalização para norma L2 = 1
   print(v);  
   print("vetor criado: ", v, end="; "); print("vectorOutRef.l2Norm=", np.linalg.norm(v));
   return v

    
def main():
    print("BD, hiperwalk/examples/coined/diagonal-grid.py data) {\n");
    print("BD, diagonal-grid, dim=3, Grover coin, numArcs = 16, nnz = 36 ) {\n");

    np.set_printoptions(linewidth=200)
    hb.init_engine(hb.CPU,0)

    initial_state = np.array( [0., 0., 0.,  0.,  0.,  0.,  0.5, -0.5, -0.5,  0.5,  0.,  0.,  0.,  0.,  0., 0.])
    second_state =  np.array([-0.5, 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0. -0.5])

    indptr  = np.array([0, 4, 6, 8, 12, 14, 16, 17, 18, 19, 20, 22, 24, 28, 30, 32, 36])
    indices = np.array([6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 13, 14, 1, 2, 0, 3, 12, 15, 
                    13, 14, 1, 2, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9])
    data = np.array([0.5, 0.5, 0.5, -0.5, 1., 0., 1., 0., 0.5, 0.5, -0.5, 0.5,
                 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 0.5, -0.5,
                 0.5, 0.5, 0., 1., 0., 1., -0.5, 0.5, 0.5, 0.5])
    Uref = csr_matrix((data, indices, indptr))

    indptr    =  np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])
    indices   =  np.array([ 9, 11,  5,  8, 14,  2,  0,  3, 12, 15, 13,  1,  7, 10,  4,  6])
    data      =  np.array([ 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    S = csr_matrix((data, indices, indptr))
    S.indices = np.ascontiguousarray(S.indices, dtype=np.int64)
    S.indptr  = np.ascontiguousarray(S.indptr,  dtype=np.int64)

    #print(S.toarray())

    indptr    =  np.array([ 0, 1, 3, 5, 6, 8, 10, 14, 18, 22, 26, 28, 30, 31, 33, 35, 36])
    indices   =  np.array([ 0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 10, 11, 10, 11, 12, 13, 14, 13, 14, 15])
    data      =  np.array([ 1., 0., 1., 1., 0., 1., 0., 1., 1., 0., -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0., 1., 1., 0., 1., 0., 1., 1., 0., 1.])
    C = csr_matrix((data, indices, indptr))
    C.indices = np.ascontiguousarray(C.indices, dtype=np.int64)
    C.indptr  = np.ascontiguousarray(C.indptr,  dtype=np.int64)
    #print(C.toarray())

    reps = 3 # 10
    n = Uref.shape[0]

    hb_Smat = hb.sparse_matrix_new(n, n, hb.FLOAT)
    hb.smatrix_connect(hb_Smat, S);
    hb.move_sparse_matrix_device(hb_Smat)

    hb_Cmat = hb.sparse_matrix_new(n, n, hb.FLOAT)
    hb.smatrix_connect(hb_Cmat, C);
    hb.move_sparse_matrix_device(hb_Cmat)

    U = C.copy()
    U.indices = np.ascontiguousarray(U.indices, dtype=np.int64)
    U.indptr  = np.ascontiguousarray(U.indptr,  dtype=np.int64)
    #print("U = C.copy(0 = \n",U.toarray())

    hb_Umat = hb.sparse_matrix_new(n, n, hb.FLOAT)
    #hb.sparse_matrix_print(hb_Umat);
    hb.smatrix_connect(hb_Umat, U);
    #hb.move_sparse_matrix_device(hb_Umat)
    hb.sparse_matrix_print(hb_Umat);
    #hb_UmatDevice = hb.load_numpy_smatrix(hb_Umat)

    hb.permute_sparse_matrix(hb_Smat, hb_Cmat, hb_Umat)
    #hb.sparse_matrix_print(hb_Umat);
    #print("U    = \n",U.toarray())
    #print("Uref = \n",Uref.toarray())
 
    np_vIn  = initial_state
    hb_vIn  = hb.vector_new(n, hb.FLOAT)
    hb.vector_connect(hb_vIn, np_vIn)
    print("++ input  vector", end=", "); hb.print_vectorT(hb_vIn)
    #hb_vIn  = hb.load_numpy_array(np_vIn)
    hb.move_vector_device(hb_vIn)
    print("++ input  vector", end=", "); hb.print_vectorT(hb_vIn)
    hb.move_vector_host(hb_vIn);
    print("++ input  vector", end=", "); hb.print_vectorT(hb_vIn)
    hb.move_vector_device(hb_vIn)
    print("++ input  vector", end=", "); hb.print_vectorT(hb_vIn)

    np_vOut = np.zeros(n)
    hb_vOut  = hb.vector_new(n, hb.FLOAT)
    hb.vector_connect(hb_vOut, np_vOut)
    #hb_vOut = hb.load_numpy_array(np_vOut)
    hb.move_vector_device(hb_vOut)

    for i in [1, 2, 3]:
       print("\n +++ new iteration, i =  ", i)
       print("++ input  vector", end=", "); hb.print_vectorT(hb_vIn)

       hb.sparse_matvec_mul(hb_Umat, hb_vIn, hb_vOut) 

       #hb.move_vector_host       (hb_vOut) # tras do device para o host
       print("++ output vector", end=", "); hb.print_vectorT(hb_vOut)

#       print("input  vector reference: ", np_vIn, end="; "); print("np_vIn.l2Norm=", np.linalg.norm(np_vIn));
       np_vOut_ref=Uref@np_vIn
#       print("output vector reference: ", np_vOut_ref, end="; "); print("np_vOut_ref.l2Norm=", np.linalg.norm(np_vOut_ref));
       print("(np_vOut_ref - np_vOut).l2Norm=", np.linalg.norm(np_vOut_ref-np_vOut));

       hb_vIn, hb_vOut = hb_vOut, hb_vIn
       np_vIn, np_vOut = np_vOut, np_vIn

    print()
    print("test complete")
    hb.stop_engine()

    return

    # comparison_result = np.allclose(result, comparison_matrix)
    
    # Print the comparison result
    # if comparison_result:
    #     print("The calculated result is the same as the comparison matrix.")
    # else:
    #     print("The calculated result is different from the comparison matrix.")

main()
