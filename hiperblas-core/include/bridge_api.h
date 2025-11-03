#ifndef __BRIDGE_API_
#define __BRIDGE_API_

#ifdef	__cplusplus
extern "C" {
#endif


typedef struct __bridge_t {

    void *plugin_handle;
    void (*InitEngine_f)(int device) ;
    void (*StopEngine_f)() ;
    long (*get_Engine_Max_Memory_Allocation_f)() ;
    void (*luDecomp_f)( void* ADev, int n );
    
    //double * (*addVector_f)(double * v1, double * v2, int n );
    vector_t * (*vector_new)( int len, data_type type, int initialize, void * data );
    void (*vector_delete)( vector_t * v ) ;
    void (*vecreqhost)( vector_t * v );
    void (*vecreqdev) ( vector_t * v );
    
    list_t * (*list_new)(); 
    list_t * (*list_append)( list_t * L, object_t o ) ;
    int (*list_len)( list_t * L ) ;
    object_t (*list_get)( list_t * L, int i) ;

    complex_t * (*complex_new)( double real, double imaginary ) ;
    void (*complex_delete)( complex_t * c ) ;

    matrix_t * (*matrix_new)( int nrow, int ncol, data_type type, int initialize, void * data );
    void (*matrix_delete)( matrix_t * v );
    void (*matreqhost)( matrix_t * v );
    void (*matreqdev) ( matrix_t * v );
    void (*matrix_set_real_value)(matrix_t *  m, int i, int j, double r);
    double (*matrix_get_real_value)(matrix_t *  m, int i, int j);
    void (*matrix_set_complex_value)(matrix_t *  m, int i, int j, double r, double im);
    double (*matrix_get_complex_real_value)(matrix_t *  m, int i, int j);
    double (*matrix_get_complex_imaginary_value)(matrix_t *  m, int i, int j);
    double * (*matrix_copy_col)(matrix_t * m, int j, int ini, int size);
    double * (*matrix_copy_row)(matrix_t * m, int i, int ini, int size);
    
    smatrix_t * (*smatrix_new)( int nrow, int ncol, data_type type );
    void (*smatrixConnect)(smatrix_t * smatrix_, double *data_, int *indptr_, int * indices_, int nnz_); //BD
    void (*smatrix_t_clear)( smatrix_t * m );
    void (*smatrix_load_double)( smatrix_t * m, FILE * f );

    void (*smatrix_set_real_value)(smatrix_t *  m, int i, int j, double r);
    void (*smatrix_pack)(smatrix_t * m);
    void (*smatrix_set_complex_value)(smatrix_t *  m, int i, int j, double r, double im);
    void (*smatrix_pack_complex)(smatrix_t * m);

    void (*smatrix_load_complex)( smatrix_t * m, FILE * f );
    void (*smatrix_delete)( smatrix_t * v );
    void (*smatreqhost)( smatrix_t * v ) ;
    void (*smatreqdev) ( smatrix_t * v );

    slist * (*slist_add)( slist * l, int col, double re, double im );
    void (*slist_clear)( slist * l );



    void * (*copyVectorFromDevice_f)(void * vec, int n );
    void * (*addVectorF_f)(void * v1Dev, void * v2Dev, int n ); 
    void * (*addVectorC_f)(void * v1Dev, void * v2Dev, int n );
    void * (*addVectorFC_f)(void * v1Dev, void * v2Dev, int n );
    
    void * (*prodVector_f)(void * v1Dev, void * v2Dev, int n ); 
    void * (*vecConj_f)(void * v1Dev, int n ); 
    void * (*vecConjugate_f)(void * v1Dev, int n ); 
    void * (*vecAddOff2_f)(void * v1Dev, int n ); 
    void * (*vecAddOff_f)(void * v1Dev, int off, int parts ); 
    void * (*prodComplexVector_f)(void * v1Dev, void * v2Dev, int n ); 
    void * (*subVector_f)(void * v1Dev, void * v2Dev, int n ); 
    void * (*subVectorC_f)(void * v1Dev, void * v2Dev, int n ); 
    void * (*mulScalarVector_f)( void * v1Dev, double scalar, int n ); 
    void * (*mulComplexScalarVector_f)( void * v1Dev, double real, double imaginary, int n ); 
    void * (*mulComplexScalarComplexVector_f)( void * v1Dev, double real, double imaginary, int n ); 
    void * (*mulFloatScalarComplexVector_f)( void * v1Dev, double real, int n ); 
    void (*mulScalarMatRow_f)( void * m, double scalar, int nrow, int ncols, int row); 
    void (*mulScalarMatCol_f)( void * m, double scalar, int nrow, int ncols, int col); 
    void * (*matVecMul1_f)(  void * mDev, void * vDev, int ncols, int nrows ); 
    void * (*matVecMul2_f)(  void * mDev, void * vDev, int ncols, int nrows ); 
    void * (*matMul_f)(  void * m1Dev, void * m2Dev, int nrows, int ncols, int qq, int atype, int btype ); 
    void * (*matMul2_f)(  void * m1Dev, void * m2Dev, int nrows, int ncols, int qq ); 
    void (*matSquare_f)( void * * outLin, void * * idxOutLin, 
                void * * outCol, void * * idxOutCol, 
                void * mLin, void * idxLin, 
                void * mCol, void * idxCol, 
                int maxcols, int N ); 
    void * (*matVecMul3_f)(  void * mDev, void * vDev, int ncols, int nrows ); 
    //void * (*sparseVecMul_f)(void * mDev, void * idxCol, void * vDev, int nrows, int maxCols ); 
    //void * (*sparseComplexVecMul_f)(void * mDev, void * idxCol, void * vDev, int nrows, int maxCols ); 
    //[Hiago]
    void * (*sparseVecMul_f)(void* v, void* m_values, void* m_row_ptr, void* m_col_idx, int m_nrows, int nnz ); 
    void * (*sparseComplexVecMul_f)(void* v, void* m_values, void* m_row_ptr, void* m_col_idx, int m_nrows, int nnz );
    //(void* mDev, void* idxCol, void* vDev, int nrows, int nnz )
    void *  (*print_smatrix_f) (const smatrix_t* matrix); //[Hiago]
    void *  (*print_vectorT_f) (const vector_t* v); //[Bidu]
   
    void * (*matVecMul3Complex_f)(  void * mDev, void * vDev, int ncols, int nrows ); 
    void * (*matTranspose_f)(  void * mDev, int ncols, int nrows ); 
    double (*sumVector_f)( void * vDev, int len ); 
    double (*normVector_f)( void * vDev, int len ); 
    double (*dotVector_f)(void * v1Dev, void * v2Dev, int len ); 
    void (*dotVectorComplex_f)( double * out_re, double * out_im,  void * v1Dev, void * v2Dev, int len ); 
} bridge_t;

typedef struct __bridge_manager_t {
    bridge_t bridges[256];  
} bridge_manager_t;




//typedef void * rmatVecMul3Complex_f(  rmatrix_t * M, void * vDev, int ncols, int nrows ); 
//rmatVecMul3Complex_f rmatVecMul3Complex; 
#ifdef	__cplusplus
}
#endif
#endif
