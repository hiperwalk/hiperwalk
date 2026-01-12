#include "gtest/gtest.h"
#include "libhiperblas.h"
#include "bridge_api.h"
#include "hiperblas_std.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>

using namespace std;

class NeblinaCoreFixture : public ::testing::Test {
protected:
public:
    bridge_manager_t m;
    int idx;

    NeblinaCoreFixture() {
        //string plugin_name = "/usr/local/lib64/libhiperblas-cpu-bridge.so";

        idx = 0;
        const char *home = getenv("HOME");
        char  *plugin_name  = (char*) malloc ( 1024 *sizeof(char));
        snprintf(plugin_name, 1024, "%s%s", home,"/hiperblas/lib/libhiperblas-cpu-bridge.so");
        printf("plugin_name = %s\n", plugin_name);

        //load_plugin(&m, const_cast<char *>(plugin_name.c_str()), idx);
        load_plugin(&m, plugin_name, idx);
        m.bridges[idx].InitEngine_f(idx); 



    }

    protected:
    static void SetUpTestSuite() {
        std::cerr << "TestSuiteSetup" << std::endl;
    }

    static void TearDownTestSuite() {
    }
    
    void SetUp() {
        
    }

    void TearDown() {
        //printf("check");
        //getchar();
        
    }

    ~NeblinaCoreFixture() {
        // cleanup any pending stuff, but no exceptions allowed
    }

};

TEST_F(NeblinaCoreFixture, vec_add) {

    int n = 3;

    vector_t * a = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    vector_t * b = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    vector_t * r;

    for (int i = 0; i < a->len; i++) {
        a->value.f[i] = 1.;
        b->value.f[i] = 1.;
    }

    object_t ** in = convertToObject(a, b);

    r = (vector_t *) vec_add(&m, idx, (void **) in, NULL);

    m.bridges[idx].vecreqhost(r);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(2., r->value.f[i]);
    }
    delete_object_array(in, 2);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(b);
    m.bridges[idx].vector_delete(r);
    
    EXPECT_EQ(1, 1);

}

// TEST_F(NeblinaCoreFixture, vec_len) {

//     int n = 3;

//     vector_t * a = m.bridges[idx].vector_new(n, T_FLOAT, 1 );

//     for (int i = 0; i < a->len; i++) {
//         a->value.f[i] = 1.;
//     }

//     object_t ** in = convertToObject(a, NULL);

//     int len = vec_len(&m, idx, (void **) in, NULL);

//     EXPECT_EQ(3, len);
//     delete_object_array(in, 1);
//     m.bridges[idx].vector_delete(a);

// }

TEST_F(NeblinaCoreFixture, vec_add_complex) {

    int n = 3;

    vector_t * a = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    vector_t * b = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    vector_t * r;

    for (int i = 0; i < a->len; i++) {
        int idx = 2 * (i);
        a->value.f[idx] = 1.;
        a->value.f[idx + 1] = 1.;
        
        b->value.f[idx] = 1.;
        b->value.f[idx + 1] = 1.;
    }

    object_t ** in = convertToObject(a, b);

    r = (vector_t *) vec_add(&m, idx, (void **) in, NULL);

    m.bridges[idx].vecreqhost(r);

    for (int i = 0; i < n; ++i) {
        int idx = 2 * (i);
        EXPECT_EQ(2., r->value.f[idx]);
        EXPECT_EQ(2., r->value.f[idx + 1]);
    }
    
    delete_object_array(in, 2);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(b);
    m.bridges[idx].vector_delete(r);

}

/*
TEST_F(NeblinaCoreFixture, vec_sub) {

    int n = 3;

    vector_t * a = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    vector_t * b = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    vector_t * r;

    for (int i = 0; i < a->len; i++) {
        a->value.f[i] = 1.;
        b->value.f[i] = 1.;
    }

    object_t ** in = convertToObject(a, b);

    r = (vector_t *) vec_sub(&m, idx, (void **) in, NULL);

    m.bridges[idx].vecreqhost(r);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(0., r->value.f[i]);
    }
    delete_object_array(in, 2);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(b);
    m.bridges[idx].vector_delete(r);

}

TEST_F(NeblinaCoreFixture, vec_sub_WithComplex) {

    int n = 3;

    vector_t * a = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    vector_t * b = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    vector_t * r;

    for (int i = 0; i < a->len; i++) {
        int idx = 2 * (i);
        a->value.f[idx] = 1.;
        a->value.f[idx + 1] = 1.;
        
        b->value.f[idx] = 1.;
        b->value.f[idx + 1] = 1.;
    }

    object_t ** in = convertToObject(a, b);

    r = (vector_t *) vec_sub(&m, idx, (void **) in, NULL);

    m.bridges[idx].vecreqhost(r);

    for (int i = 0; i < n; ++i) {
        int idx = 2 * (i);
        EXPECT_EQ(0., r->value.f[idx]);
        EXPECT_EQ(0., r->value.f[idx + 1]);
    }

    delete_object_array(in, 2);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(b);
    m.bridges[idx].vector_delete(r);

}


TEST_F(NeblinaCoreFixture, scalar_vec) {

    int n = 3;
    
    double scalar = 2.0;
    vector_t * a = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    
    for (int i = 0; i < a->len; i++) {
        a->value.f[i] = 2.;
    }

    object_t ** in = convertScaVecToObject(scalar, a);
    
    vector_t * r = (vector_t *) vec_mulsc(&m, idx, (void **) in, NULL);
    
    m.bridges[idx].vecreqhost(r);
    
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(4., r->value.f[i]);
    }

    delete_object_array(in, 2);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(r);
}

TEST_F(NeblinaCoreFixture, complex_scalar_float_vec) {

    int n = 3;
    
    complex_t * scalar = m.bridges[idx].complex_new(2.0, 2.0);
    vector_t * a = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    
    for (int i = 0; i < a->len; i++) {
        a->value.f[i] = 2.;
    }

    vector_t * r = (vector_t *) vec_mul_complex_scalar(&m, idx, scalar, a);
    
    m.bridges[idx].vecreqhost(r);
    
    for (int i = 0; i < r->len; i++) {
        int idx = 2 * (i);

        EXPECT_EQ(4., r->value.f[idx]);
        EXPECT_EQ(2., r->value.f[idx + 1]);
    }

    m.bridges[idx].vector_delete(a);
    m.bridges[idx].complex_delete(scalar);
    m.bridges[idx].vector_delete(r);
}

TEST_F(NeblinaCoreFixture, complex_scalar_complex_vec) {

    int n = 3;
    
    complex_t * scalar = m.bridges[idx].complex_new(2.0, 2.0);
    vector_t * a = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    
    for (int i = 0; i < a->len; i++) {
        int idx = 2 * (i);
        a->value.f[idx] = 2.;
        a->value.f[idx+1] = 2.;
    }

    vector_t * r = (vector_t *) mul_complex_scalar_complex_vec(&m, idx, scalar, a);
    
    m.bridges[idx].vecreqhost(r);
    
    for (int i = 0; i < r->len; i++) {
        int idx = 2 * (i);

        EXPECT_EQ(4., r->value.f[idx]);
        EXPECT_EQ(4., r->value.f[idx + 1]);
    }

    m.bridges[idx].vector_delete(a);
    m.bridges[idx].complex_delete(scalar);
    m.bridges[idx].vector_delete(r);
}

TEST_F(NeblinaCoreFixture, float_scalar_complex_vec) {

    int n = 3;
    
    double scalar = 2.0;
    vector_t * a = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    
    for (int i = 0; i < a->len; i++) {
        int idx = 2 * (i);
        a->value.f[idx] = 2.;
        a->value.f[idx+1] = 2.;
    }

    vector_t * r = (vector_t *) mul_float_scalar_complex_vec(&m, idx, scalar, a);
    
    m.bridges[idx].vecreqhost(r);
    
    for (int i = 0; i < r->len; i++) {
        int idx = 2 * (i);

        EXPECT_EQ(4., r->value.f[idx]);
        EXPECT_EQ(2., r->value.f[idx + 1]);
    }

    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(r);
}

TEST_F(NeblinaCoreFixture, vec_prod_WithFloat) {

    int n = 3;

    vector_t * a = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    vector_t * b = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    vector_t * r;

    for (int i = 0; i < a->len; i++) {
        a->value.f[i] = 2.;
        b->value.f[i] = 2.;
    }

    object_t ** in = convertToObject(a, b);

    r = (vector_t *) vec_prod(&m, idx, (void **) in, NULL);

    m.bridges[idx].vecreqhost(r);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(4., r->value.f[i]);
    }
    delete_object_array(in, 2);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(b);
    m.bridges[idx].vector_delete(r);

}

TEST_F(NeblinaCoreFixture, vec_prod_WithComplex) {

    int n = 3;

    vector_t * a = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    vector_t * b = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    vector_t * r;

    for (int i = 0; i < 2 * a->len; i += 2) {
        a->value.f[i] = 2.;
        a->value.f[i + 1] = 2.;
        b->value.f[i] = 2.;
        b->value.f[i + 1] = 2.;
    }

    object_t ** in = convertToObject(a, b);

    r = (vector_t *) vec_prod(&m, idx, (void **) in, NULL);

    m.bridges[idx].vecreqhost(r);

    for (int i = 0; i < 2 * a->len; i += 2) {
        EXPECT_EQ(0., r->value.f[i]);
        EXPECT_EQ(8., r->value.f[i + 1]);
    }
    delete_object_array(in, 2);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(b);
    m.bridges[idx].vector_delete(r);

}

TEST_F(NeblinaCoreFixture, vec_conj) {

    int n = 3;

    vector_t * a = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    vector_t * r;

    for (int i = 0; i < 2 * a->len; i += 2) {
        a->value.f[i] = 2.;
        a->value.f[i + 1] = 2.;
    }

    object_t ** in = convertToObject(a, NULL);

    r = (vector_t *) vec_conj(&m, idx, (void **) in, NULL);

    m.bridges[idx].vecreqhost(r);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(2., r->value.f[2 * i]);
        EXPECT_EQ(-2., r->value.f[2 * i + 1]);
    }
    delete_object_array(in, 1);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(r);

}

TEST_F(NeblinaCoreFixture, vec_add_off) {

    int n = 4;

    vector_t * a = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    vector_t * r;

    for (int i = 0; i < a->len; i++) {
        a->value.f[i] = 2.;
    }
    int offset = 2;
    object_t ** in = convertToObject2(offset, a);

    r = (vector_t *) vec_add_off(&m, idx, (void **) in, NULL);

    m.bridges[idx].vecreqhost(r);

    for (int i = 0; i < offset; ++i) {
        EXPECT_EQ(4., r->value.f[i]);
    }
    delete_object_array(in, 2);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(r);

}

TEST_F(NeblinaCoreFixture, vec_sum) {

    int n = 4;

    vector_t * a = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    object_t * r;

    for (int i = 0; i < a->len; i++) {
        a->value.f[i] = 2.;
    }

    // printf("vec_sum 1\n");
    object_t ** in = convertToObject(a, NULL);
    // printf("vec_sum 2\n");

    r = (object_t *) vec_sum(&m, idx, (void **) in, NULL);
    // printf("vec_sum 3\n");

    EXPECT_EQ(8., r->value.f);
    // printf("vec_sum 4\n");
    
    delete_object_array(in, 1);
    delete_object(r);
    m.bridges[idx].vector_delete(a);
    // printf("vec_sum 5\n");

}

TEST_F(NeblinaCoreFixture, addVectorC) {

    int n = 3;

    vector_t * a = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    vector_t * b = m.bridges[idx].vector_new(n, T_COMPLEX, 1, NULL );
    vector_t * r;


    for (int i = 0; i < 2 * a->len; i += 2) {
        a->value.f[i] = 1.;
        a->value.f[i + 1] = 1.;
        b->value.f[i] = 1.;
        b->value.f[i + 1] = 1.;
    }

    object_t ** in = convertToObject(a, b);

    r = (vector_t *) vec_add(&m, idx, (void **) in, NULL);
    
    m.bridges[idx].vecreqhost(r);

    for (int i = 0; i < 2 * a->len; i += 2) {
        EXPECT_EQ(2., r->value.f[i]);
        EXPECT_EQ(2., r->value.f[i + 1]);
    }
    delete_object_array(in, 2);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(b);
    m.bridges[idx].vector_delete(r);
}

TEST_F(NeblinaCoreFixture, subVector) {

    int n = 3;

    vector_t * a = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );
    vector_t * b = m.bridges[idx].vector_new(n, T_FLOAT, 1, NULL );

    vector_t * r;


    for (int i = 0; i < a->len; i++) {
        a->value.f[i] = 1.;
        b->value.f[i] = 1.;
    }

    object_t ** in = convertToObject(a, b);

    r = (vector_t *) vec_sub(&m, idx, (void **) in, NULL);

    m.bridges[idx].vecreqhost(r);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(0., r->value.f[i]);
    }
    delete_object_array(in, 2);
    m.bridges[idx].vector_delete(a);
    m.bridges[idx].vector_delete(b);
    m.bridges[idx].vector_delete(r);

}
*/
