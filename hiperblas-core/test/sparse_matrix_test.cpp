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
        plugin_name = "//home/bidu/hiperblas/lib/libhiperblas-cpu-bridge.so";
        plugin_name = "/mnt/c/Users/bidu/OneDrive/aLncc/passeiosQuantHiago/Bhiperblas-core-old/libhiperblas-cpu-bridge.so";
        plugin_name = "/prj/prjedlg/bidu/hiperblas/lib/libhiperblas-cpu-bridge.so";
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

    int n = 4;

    vector_t  * a = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    smatrix_t * b = m.bridges[idx].smatrix_new(n, n, T_FLOAT);
    vector_t * r;

    for (int i = 0; i < a->len; i++) { a->value.f[i] = 1.; }
    m.bridges[idx].print_vectorT_f(a);

    printf("BD, em TEST_F, vetor  entrada \n");
    for (int i = 0; i < a->len; i++) { printf("r[%2d]=%f\n", i,  a->value.f[i]); }

    m.bridges[idx].smatrix_set_real_value(b, 0, 0, 1.);
    m.bridges[idx].smatrix_set_real_value(b, 0, 1, 1.);
//    m.bridges[idx].smatrix_set_real_value(b, 0, 2, 3.);
//    m.bridges[idx].smatrix_set_real_value(b, 0, 3, 3.);
    m.bridges[idx].smatrix_set_real_value(b, 1, 0, 1.);
    m.bridges[idx].smatrix_set_real_value(b, 1, 1, 1.);
 //   m.bridges[idx].smatrix_set_real_value(b, 1, 2, 3.);
//    m.bridges[idx].smatrix_set_real_value(b, 1, 3, 3.);
 //   m.bridges[idx].smatrix_set_real_value(b, 2, 0, 3.);
  //  m.bridges[idx].smatrix_set_real_value(b, 2, 1, 3.);
    m.bridges[idx].smatrix_set_real_value(b, 2, 2, 2.);
    m.bridges[idx].smatrix_set_real_value(b, 2, 3, 2.);
   // m.bridges[idx].smatrix_set_real_value(b, 2, 4, 3.);
//    m.bridges[idx].smatrix_set_real_value(b, 3, 0, 3.);
//    m.bridges[idx].smatrix_set_real_value(b, 3, 1, 3.);
    m.bridges[idx].smatrix_set_real_value(b, 3, 2, 2.);
    m.bridges[idx].smatrix_set_real_value(b, 3, 3, 2.);
    //m.bridges[idx].smatrix_set_real_value(b, 3, 4, 3.);
    //m.bridges[idx].smatrix_set_real_value(b, 4, 2, 3.);
    //m.bridges[idx].smatrix_set_real_value(b, 4, 3, 3.);
    //m.bridges[idx].smatrix_set_real_value(b, 4, 4, 3.);

    m.bridges[idx].smatrix_pack(b);
    m.bridges[idx].print_smatrix_f(b);

    m.bridges[idx].vecreqdev(a);
    m.bridges[idx].smatreqdev(b);

    object_t ** in = convertToObject4(a, b);

    //printf("BD, em TEST_F, before call matvec_mul3\n");
    r = (vector_t *) matvec_mul3(&m, idx, (void **) in, NULL);
    m.bridges[idx].vecreqhost(r);
    printf("BD, em TEST_F, after  call matvec_mul3\n");
    m.bridges[idx].print_vectorT_f(r);
    
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

