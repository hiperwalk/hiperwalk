#include "gtest/gtest.h"
#include "libhiperblas.h"
#include "bridge_api.h"
#include "hiperblas_std.h"
//#include "hiperblas_vector.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <string>

using namespace std;

class SparseMatrixFixture : public ::testing::Test {
protected:
public:
    bridge_manager_t m;
    int idx;
    
    SparseMatrixFixture() {
        idx = 0;
        string plugin_name = "/usr/local/lib64/libhiperblas-cpu-bridge.so";
        plugin_name = "/mnt/c/Users/bidu/OneDrive/aLncc/passeiosQuantHiago/Bhiperblas-core-old/libhiperblas-cpu-bridge.so";
        plugin_name = "/prj/prjedlg/bidu/hiperblas/lib/libhiperblas-cpu-bridge.so";
        plugin_name = "/home/bidu/hiperblas/lib/libhiperblas-cpu-bridge.so";
        load_plugin(&m, const_cast<char *>(plugin_name.c_str()), idx);
        m.bridges[idx].InitEngine_f(0);
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

    printf("\nBD, inicio de TEST_F(SparseMatrixFixture, matvec_mul3_WithSparseMatrixFloat) {\n");
    printf("\nBD, ./examples/coined/diagonal-grid.py data) {\n");
    printf("\nBD, diagonal-gridStencil, dim=3, Grover coin, numArcs = 16, nnz = 36 ) {\n");

    int n = 16;
    vector_t  * a = m.bridges[idx].vector_new (n, T_FLOAT, 1, NULL );
    smatrix_t * b = m.bridges[idx].smatrix_new(n, n, T_FLOAT);
    b->nnz=36;
    vector_t  * r = m.bridges[idx].vector_new (n, T_FLOAT, 1, NULL );
 
 
//initial state = [ 0.000  0.000  0.000  0.000  0.000  0.000  0.500 -0.500 -0.500  0.500  0.000  0.000  0.000  0.000  0 .000  0.000];  state.l2Norm= 1.0
    int i;
    for (i = 0; i < a->len; i++) { a->value.f[i] = 0.; }
    i=6; a->value.f[i] =  0.500; i=7; a->value.f[i] = -0.500;
    i=8; a->value.f[i] = -0.500; i=9; a->value.f[i] =  0.500;
    a->extra=a->value.f;
    printf("BD, em TEST_F, vetor  entrada \n");
    m.bridges[idx].print_vectorT_f(a);

//v_->extra [0:15]: -0.50 0.00 0.00 0.50 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.50 0.00 0.00 -0.50, L2Norm = 1.000000
    for (i = 0; i < r->len; i++) { r->value.f[i] = 0.; }
    i=0;  r->value.f[i] = -0.500; i=3; r->value.f[i] =  0.500;
    i=12; r->value.f[i] = 0.500; i=15; r->value.f[i] = -0.500;
    r->extra=r->value.f;
    printf("BD, em TEST_F, vetor  saida \n");
    m.bridges[idx].print_vectorT_f(r);

    //for (int i = 0; i < a->len; i++) { printf("r[%2d]=%f\n", i,  a->value.f[i]); }

//U.indptr    =  [ 0  4  6  8 12 14 16 17 18 19 20 22 24 28 30 32 36]
    b->row_ptr = (long long int*) malloc((n+1)*sizeof(long long int)) ;
    i=0;
    b->row_ptr[i++]=0;   b->row_ptr[i++]=4;   b->row_ptr[i++]=6;   b->row_ptr[i++]=8;
    b->row_ptr[i++]=12;  b->row_ptr[i++]=14;  b->row_ptr[i++]=16;  b->row_ptr[i++]=17;
    b->row_ptr[i++]=18;  b->row_ptr[i++]=19;  b->row_ptr[i++]=20;  b->row_ptr[i++]=22;
    b->row_ptr[i++]=24;  b->row_ptr[i++]=28;  b->row_ptr[i++]=30;  b->row_ptr[i++]=32;
    b->row_ptr[i++]=36;
//U.indices   =  [ 6  7  8  9 10 11  4  5  6  7  8  9 13 14  1  2  0  3 12 15 13 14  1  2  6  7  8  9 10 11  4  5  6  7  8  9]
    b->col_idx = (long long int*) malloc((b->nnz)*sizeof(long long int)) ;
    i=0;
    b->col_idx[i++]=6;   b->col_idx[i++]=7;   b->col_idx[i++]=8;   b->col_idx[i++]=9;
    b->col_idx[i++]=10;   b->col_idx[i++]=11; 
    b->col_idx[i++]=4;   b->col_idx[i++]=5;
    b->col_idx[i++]=6;   b->col_idx[i++]=7;   b->col_idx[i++]=8;   b->col_idx[i++]=9;
    b->col_idx[i++]=13;   b->col_idx[i++]=14;
    b->col_idx[i++]=1;   b->col_idx[i++]=2;
    b->col_idx[i++]=0;
    b->col_idx[i++]=3;
    b->col_idx[i++]=12;
    b->col_idx[i++]=15;
    b->col_idx[i++]=13;   b->col_idx[i++]=14;
    b->col_idx[i++]=1;   b->col_idx[i++]=2;
    b->col_idx[i++]=6;   b->col_idx[i++]=7;   b->col_idx[i++]=8;   b->col_idx[i++]=9;
    b->col_idx[i++]=10;   b->col_idx[i++]=11; 
    b->col_idx[i++]=4;   b->col_idx[i++]=5;
    b->col_idx[i++]=6;   b->col_idx[i++]=7;   b->col_idx[i++]=8;   b->col_idx[i++]=9;

//U.data      =  [ 0.5  0.5  0.5 -0.5  1.   0.   1.   0.   0.5  0.5 -0.5  0.5  1.   0.   1.   0.   1.   1.   1.   1.   0.   1.   0.   1.   0.5 -0.5  0.5  0.5  0.   1.   0.   1.  -0.5  0.5  0.5  0.5]
    b->values= (double*) malloc((b->nnz)*sizeof(double)) ;
    i=0;
    b->values[i++]=0.5;   b->values[i++]=0.5;   b->values[i++]=0.5;   b->values[i++]=-0.5;
    b->values[i++]=1.0;   b->values[i++]=0.0; 
    b->values[i++]=1.0;   b->values[i++]=0.0; 
    b->values[i++]=0.5;   b->values[i++]=0.5;   b->values[i++]=-0.5;   b->values[i++]=0.5;
    b->values[i++]=1.0;   b->values[i++]=0.0; 
    b->values[i++]=1.0;   b->values[i++]=0.0; 

    b->values[i++]=1.0; 
    b->values[i++]=1.0; 
    b->values[i++]=1.0; 
    b->values[i++]=1.0; 
    b->values[i++]=0.0;   b->values[i++]=1.0; 
    b->values[i++]=0.0;   b->values[i++]=1.0; 
    b->values[i++]=0.5;   b->values[i++]=-0.5;   b->values[i++]=0.5;   b->values[i++]=0.5;
    b->values[i++]=0.0;   b->values[i++]=1.0; 
    b->values[i++]=0.0;   b->values[i++]=1.0; 
    b->values[i++]=-0.5;   b->values[i++]=0.5;   b->values[i++]=0.5;   b->values[i++]=0.5;
    printf("BD, em TEST_F, matriz U \n");
    m.bridges[idx].print_smatrix_f(b);

    //m.bridges[idx].smatrix_pack(b);

    m.bridges[idx].vecreqdev(a);
    m.bridges[idx].smatreqdev(b);

    object_t ** in = convertToObject4(a, b);

    //printf("BD, em TEST_F, before call matvec_mul3\n");
    //r = (vector_t *) matvec_mul3BD(&m, idx, (void **) in, NULL);
    matvec_mul3BD(&m, idx, (void **) in, NULL);
    vector_t  * rOut = (vector_t *) in[2];
    m.bridges[idx].vecreqhost(rOut);
    printf("BD, em TEST_F, after  call matvec_mul3\n");
    m.bridges[idx].print_vectorT_f(rOut);
    
    //printf("BD, em TEST_F, vetor resultado apos:  call matvec_mul3\n");
    //for (int i = 0; i < r->len; i++) { printf("r[%2d]=%f\n", i,  r->value.f[i]); }

    printf("BD, final  de TEST_F(SparseMatrixFixture, matvec_mul3_WithSparseMatrixFloat) {\n");
    printf("meu exit(2222)\n"); exit(2222);

    int tamBlocos=2;
    for (int i = 0; i < r->len; i++) {
    EXPECT_EQ(3.0*tamBlocos, r->value.f[i]);
       }

    return;

    EXPECT_EQ(27., r->value.f[1]);
    EXPECT_EQ(27., r->value.f[2]);
    EXPECT_EQ(27., r->value.f[3]);
    EXPECT_EQ(0., r->value.f[4]);
    EXPECT_EQ(0., r->value.f[5]);
    EXPECT_EQ(0., r->value.f[6]);
    EXPECT_EQ(0., r->value.f[7]);
    EXPECT_EQ(0., r->value.f[8]);
    EXPECT_EQ(0., r->value.f[9]);

    delete_object_array(in, 2);

    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(r);
    m.bridges[idx].smatrix_delete(b);

}

