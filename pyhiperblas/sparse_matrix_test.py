import numpy as np
import pytest
from scipy.sparse import random
from scipy.sparse import block_diag, csr_matrix
import hiperblas as hb

def csr_bloco_diagonal(blocos):
    """
    Cria uma matriz CSR bloco-diagonal a partir de uma lista de blocos.
    Cada bloco pode ser uma matriz densa (numpy) ou esparsa (scipy).
    """
    return block_diag(blocos, format='csr')


def vector_normL2_one (n_):
   v = np.random.randn(n_)    # vetor aleatório
   v = v / np.linalg.norm(v) # normalização para norma L2 = 1
   print(v);  
   print("vetor criado: ", v, end="; "); print("vectorOutRef.l2Norm=", np.linalg.norm(v));
   return v


    
def main():

    hb.init_engine(hb.CPU,0)

    reps = 1 # 10

    blA = np.array([[1, 2], [3, 4]])
    blB = np.array([[5]])
    blC = np.array([[6, 7, 8], [7, 7, 6], [8, 6, 8]])
    blocoDiagMat = csr_bloco_diagonal([blA, blB, blC])

    n = blocoDiagMat.shape[0]
    float_vectorIn  = vector_normL2_one(n) #  np.ones(n)
    float_vectorOut = np.zeros(n)

    print(blocoDiagMat.toarray())

    hb_mat = hb.sparse_matrix_new(n, n, hb.FLOAT)

    hb.sparse_matrix_print(hb_mat);
    hb.smatrixConnect(hb_mat, blocoDiagMat);
    hb.sparse_matrix_print(hb_mat);

    hb_vIn  = hb.load_numpy_array(float_vectorIn)
    hb.move_vector_device(hb_vIn)
    hb.print_vectorT(hb_vIn)

    hb_vOut = hb.load_numpy_array(float_vectorOut)
    hb.move_vector_device(hb_vOut)
    hb.print_vectorT(hb_vOut)

    hb.sparse_matvec_mul(hb_mat, hb_vIn, hb_vOut) 
    hb.print_vectorT(hb_vOut)
    hb.move_vector_host       (hb_vOut) # tras do device para o host
    hb.print_vectorT(hb_vOut)

    vectorOutRef=blocoDiagMat@float_vectorIn
    print("vetor referencia: ", vectorOutRef, end="; "); print("vectorOutRef.l2Norm=", np.linalg.norm(vectorOutRef));

    #i=n//2; vectorOutRef[i] = vectorOutRef[i]+0.1 # to include a error
    print("vector output test")
    for i in range(n):
        print(f"{float_vectorOut[i] - vectorOutRef[i]:+12.6e}", end=", ")
        assert float_vectorOut[i]  == pytest.approx(vectorOutRef[i], 1e-12)

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
