#include "gtest/gtest.h"
#include "libhiperblas.h"
#include "bridge_api.h"
#include "hiperblas_std.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <string>

using namespace std;

class SparseMatrixFixture : public ::testing::Test {
protected:
public:
    bridge_manager_t bridgeManager;
    int idx;
    
    SparseMatrixFixture() {
        idx = 0;
	/*
        string plugin_name = "/usr/local/lib64/libhiperblas-cpu-bridge.so";
        plugin_name = "/mnt/c/Users/bidu/OneDrive/aLncc/passeiosQuantHiago/Bhiperblas-core-old/libhiperblas-cpu-bridge.so";
        plugin_name = "/home/bidu/hiperblas/lib/libhiperblas-cpu-bridge.so";
        plugin_name = "/prj/prjedlg/bidu/hiperblas/lib/libhiperblas-cpu-bridge.so";
        plugin_name = "/prj/cenapadrjsd/eduardo.garcia2/hiperblas/lib/libhiperblas-cpu-bridge.so";
	*/

        const char *home = getenv("HOME");
        char  *plugin_name  = (char*) malloc ( 1024 *sizeof(char));
 //  compondo o nome absoluto do arquivo de biblioteca: plugin_nam= home+"/hiperbl...-cpu-bridge.so"
        snprintf(plugin_name, 1024, "%s%s", home,"/local/lib/libhiperblas-cpu-bridge.so");
        printf("plugin_name = %s\n", plugin_name);

        //load_plugin(&bridgeManager, const_cast<char *>(plugin_name.c_str()), idx);
        load_plugin(&bridgeManager, plugin_name, idx);
        bridgeManager.bridges[idx].InitEngine_f(0);
    }

    protected:
    static void SetUpTestSuite() {
        std::cerr << "Sparse Matrix TestSuiteSetup" << std::endl;
    }

    static void TearDownTestSuite() {
    }
    
    void SetUp() {
        
    }

    void TearDown() {
        //printf("check");
        //getchar();
        
    }

    ~SparseMatrixFixture() {
        // cleanup any pending stuff, but no exceptions allowed
    }

};


TEST_F(SparseMatrixFixture, matvec_mul3_WithSparseMatrixFloat) {

    printf("\nbd, inicio de TEST_F(SparseMatrixFixture, matvec_mul3_WithSparseMatrixFloat) {\n");
    printf("\nbd, ./examples/coined/diagonal-grid.py data) {\n");
    printf("\nbd, diagonal-gridStencil, dim=3, Grover coin, numArcs = 16, nnz = 36 ) {\n");

    // Compressed Sparse Row Matrix
    int n = 16;
 
//initial state = [ 0.000  0.000  0.000  0.000  0.000  0.000  0.500 -0.500 -0.500  0.500  0.000  0.000  0.000  0.000  0 .000  0.000];  state.l2Norm= 1.0
    vector_t  * inV = bridgeManager.bridges[idx].vector_new (n, T_FLOAT, 1, NULL );
    int i; for (i = 0; i < inV->len; i++) { inV->value.f[i] = 0.; }
    i=6; inV->value.f[i] =  0.500; i=7; inV->value.f[i] = -0.500;
    i=8; inV->value.f[i] = -0.500; i=9; inV->value.f[i] =  0.500;
 //   a->extra=a->value.f;
    printf("bd, em TEST_F, vetor  entrada \n");
    print_vectorT(inV);

//v_->extra [0:15]: -0.50 0.00 0.00 0.50 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.50 0.00 0.00 -0.50, L2Norm = 1.000000
    vector_t  * referenceOut = bridgeManager.bridges[idx].vector_new (n, T_FLOAT, 1, NULL );
    for (i = 0; i < referenceOut->len; i++) { referenceOut->value.f[i] = 0.; }
    i=0;  referenceOut->value.f[i] = -0.500;  i=3; referenceOut->value.f[i] =  0.500;
    i=12; referenceOut->value.f[i] =  0.500; i=15; referenceOut->value.f[i] = -0.500;
    //printf("bd, em TEST_F, vetor  saida, apos uma iteracao:\n");
    //print_vectorT(referenceOut);

//U.indptr    =  [ 0  4  6  8 12 14 16 17 18 19 20 22 24 28 30 32 36]
    smatrix_t * csrM = bridgeManager.bridges[idx].smatrix_new(n, n, T_FLOAT);
    csrM->nnz=36;
    csrM->row_ptr = (long int*) malloc((n+1)*sizeof(long int)) ;
    i=0;
    csrM->row_ptr[i++]=0;   csrM->row_ptr[i++]=4;   csrM->row_ptr[i++]=6;   csrM->row_ptr[i++]=8;
    csrM->row_ptr[i++]=12;  csrM->row_ptr[i++]=14;  csrM->row_ptr[i++]=16;  csrM->row_ptr[i++]=17;
    csrM->row_ptr[i++]=18;  csrM->row_ptr[i++]=19;  csrM->row_ptr[i++]=20;  csrM->row_ptr[i++]=22;
    csrM->row_ptr[i++]=24;  csrM->row_ptr[i++]=28;  csrM->row_ptr[i++]=30;  csrM->row_ptr[i++]=32;
    csrM->row_ptr[i++]=36;
//U.indices   =  [ 6  7  8  9 10 11  4  5  6  7  8  9 13 14  1  2  0  3 12 15 13 14  1  2  6  7  8  9 10 11  4  5  6  7  8  9]
    csrM->col_idx = (long int*) malloc((csrM->nnz)*sizeof(long int)) ;
    i=0;
    csrM->col_idx[i++]=6;   csrM->col_idx[i++]=7;   csrM->col_idx[i++]=8;   csrM->col_idx[i++]=9;
    csrM->col_idx[i++]=10;  csrM->col_idx[i++]=11; 
    csrM->col_idx[i++]=4;   csrM->col_idx[i++]=5;
    csrM->col_idx[i++]=6;   csrM->col_idx[i++]=7;   csrM->col_idx[i++]=8;   csrM->col_idx[i++]=9;
    csrM->col_idx[i++]=13;  csrM->col_idx[i++]=14;
    csrM->col_idx[i++]=1;   csrM->col_idx[i++]=2;
    csrM->col_idx[i++]=0;
    csrM->col_idx[i++]=3;
    csrM->col_idx[i++]=12;
    csrM->col_idx[i++]=15;
    csrM->col_idx[i++]=13;  csrM->col_idx[i++]=14;
    csrM->col_idx[i++]=1;   csrM->col_idx[i++]=2;
    csrM->col_idx[i++]=6;   csrM->col_idx[i++]=7;   csrM->col_idx[i++]=8;   csrM->col_idx[i++]=9;
    csrM->col_idx[i++]=10;  csrM->col_idx[i++]=11; 
    csrM->col_idx[i++]=4;   csrM->col_idx[i++]=5;
    csrM->col_idx[i++]=6;   csrM->col_idx[i++]=7;   csrM->col_idx[i++]=8;   csrM->col_idx[i++]=9;

//U.data      =  [ 0.5  0.5  0.5 -0.5  1.   0.   1.   0.   0.5  0.5 -0.5  0.5  1.   0.   1.   0.   1.   1.   1.   1.   0.   1.   0.   1.   0.5 -0.5  0.5  0.5  0.   1.   0.   1.  -0.5  0.5  0.5  0.5]
    csrM->values= (double*) malloc((csrM->nnz)*sizeof(double)) ;
    i=0;
    csrM->values[i++]=0.5;   csrM->values[i++]=0.5;   csrM->values[i++]=0.5;   csrM->values[i++]=-0.5;
    csrM->values[i++]=1.0;   csrM->values[i++]=0.0; 
    csrM->values[i++]=1.0;   csrM->values[i++]=0.0; 
    csrM->values[i++]=0.5;   csrM->values[i++]=0.5;   csrM->values[i++]=-0.5;   csrM->values[i++]=0.5;
    csrM->values[i++]=1.0;   csrM->values[i++]=0.0; 
    csrM->values[i++]=1.0;   csrM->values[i++]=0.0; 

    csrM->values[i++]=1.0; 
    csrM->values[i++]=1.0; 
    csrM->values[i++]=1.0; 
    csrM->values[i++]=1.0; 
    csrM->values[i++]=0.0;   csrM->values[i++]=1.0; 
    csrM->values[i++]=0.0;   csrM->values[i++]=1.0; 
    csrM->values[i++]=0.5;   csrM->values[i++]=-0.5;   csrM->values[i++]=0.5;   csrM->values[i++]=0.5;
    csrM->values[i++]=0.0;   csrM->values[i++]=1.0; 
    csrM->values[i++]=0.0;   csrM->values[i++]=1.0; 
    csrM->values[i++]=-0.5;  csrM->values[i++]=0.5;   csrM->values[i++]=0.5;   csrM->values[i++]=0.5;
    printf("bd, em TEST_F, matriz U \n");
    print_smatrix(csrM);

    bridgeManager.bridges[idx].vecreqdev (inV);  // create new vector at Dev with inV values

    vector_t  * outV = bridgeManager.bridges[idx].vector_new (n, T_FLOAT, 1, NULL ); // new vector at Dev
    bridgeManager.bridges[idx].vecreqdev (outV);
    print_vectorT(outV);

    object_t ** in3RefObj = convertToObject4BD(csrM, inV, outV ); // pack a 3 references at an object 
								
    for (int state_index=1; state_index<=2; state_index++){

       printf("bd, em TEST_F,  ..............  state_index = %d\n", state_index);
       printf("bd, em TEST_F, just before call matvec_mul3\n");
       printf("bd, em TEST_F, input  vetor \n"); print_vectorT(inV);
       matvec_mul3(&bridgeManager, idx, (void **) in3RefObj, NULL);

       printf("bd, em TEST_F, after  call matvec_mul3\n");
       bridgeManager.bridges[idx].vecreqhost(outV); // r->value.f = (double*) r->extra;
       printf("bd, em TEST_F, output vetor \n"); print_vectorT(outV);

       if(state_index==1) {
           printf("bd, em TEST_F, reference output \n"); print_vectorT(referenceOut);
           printf("bd, verificação do resultado da função matvec_mul3:");
           for (int i = 0; i < referenceOut->len; i++) { EXPECT_EQ( outV->value.f[i], referenceOut->value.f[i]); printf("%d:...OK!, ", i); }
       }
        printf("\n");
        vector_t  * tmpV;
        tmpV = outV; outV = inV; inV = tmpV;
    }
     

       printf("bd, em TEST_F, before call matvec_mul3\n");
       matvec_mul3(&bridgeManager, idx, (void **) in3RefObj, NULL);

       printf("bd, em TEST_F, after  call matvec_mul3\n");
       bridgeManager.bridges[idx].vecreqhost(outV); // r->value.f = (double*) r->extra;
       printf("bd, em TEST_F, output vetor \n"); print_vectorT(outV);
    
    delete_object_array(in3RefObj, 2);
    bridgeManager.bridges[idx].vector_delete(inV);
    bridgeManager.bridges[idx].vector_delete(referenceOut);
    bridgeManager.bridges[idx].vector_delete(outV);
    bridgeManager.bridges[idx].smatrix_delete(csrM);
    return;

}

