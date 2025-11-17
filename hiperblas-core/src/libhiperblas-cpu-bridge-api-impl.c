#ifndef __NEBLINAVECTOROCL
#define __NEBLINAVECTOROCL
#define  BLOCK_DIM 16

#include <math.h>
#include "libhiperblas.h"
#include "hiperblas.h"
#include "bridge_api.h"

void InitEngine(int device){
    //printf("InitEngine\n");
    //InitCLEngine(device);
    //printf("end InitEngine\n");
}

void StopEngine(){
    //ReleaseCLInfo(clinfo);
}

long get_Engine_Max_Memory_Allocation(){
    return 0L;
}

void luDecomp(void* v1Dev, int n ) {
    
    // return (void *)NULL;
}

double * copyVectorFromDevice( double * vec, int n ) {
    double * out = (double *) malloc( n );
    memcpy(out, vec, n);
    return out;
}

double * addVectorF( double * v1, double * v2, int n ) {
    double * out = (double *) malloc( n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        //printf("%f %f\n",v1[i] , v2[i]);
        out[i] = v1[i] + v2[i];
    }

    return out;
}


void* addVectorC(double* v1, double* v2, int n ) {
    
    double * out = (double *) malloc( 2 * n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        out[2*i] = v1[2*i] + v2[2*i];
        out[2*i+1] = v1[2*i+1] + v2[2*i+1];
    }

    return out;
}

void* addVectorFC(double* v1, double* v2, int n ) {
    double * out = (double *) malloc( 2 * n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        out[2*i] = v1[i] + v2[2*i];
        out[2*i+1] = v2[2*i+1];
    }

    return out;
}

void* prodVector(double* v1, double* v2, int n ) {
    double * out = (double *) malloc( n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        out[i] = v1[i] * v2[i];
    }

    return out;
}

void* vecConj(void* v1Dev, int n ) {
    
    return (void *)NULL;
}

void* vecConjugate(double* v1, int n ) {
    double * out = (double *) malloc(2 * n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        out[2*i] = v1[2*i];
        out[2*i+1] = -v1[2*i+1];
    }
    return out;
}

void* vecAddOff2(void* v1Dev, int n ) {
    
    return (void *)NULL;
}

void* vecAddOff(double* v1, int offset, int parts ) {
    size_t n = parts * offset;

    double * out = (double *) malloc( offset * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i < offset; i++) {
        double s = 0;
        for(int l=0; l < parts; l++ ) {
            int idx = i + l * offset;
            s += v1[idx];   
        }
        out[i] = s;
    }
    return out;
}

void* prodComplexVector(double* v1, double* v2, int n ) {
    
    double * out = (double *) malloc(2 * n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        int idx_re = 2*i;
        int idx_im = 2*i+1;
        out[idx_re]  = v1[idx_re] * v2[idx_re] - v1[idx_im] * v2[idx_im];
        out[idx_im]  = v1[idx_re] * v2[idx_im] + v1[idx_im] * v2[idx_re];
    }

    return out;
}

void* subVector(double* v1, double* v2, int n ) {
    double * out = (double *) malloc( n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        //printf("%f %f\n",v1[i] , v2[i]);
        out[i] = v1[i] - v2[i];
    }

    return out;
}

void* subVectorC(double* v1, double* v2, int n ) {
    
    double * out = (double *) malloc( 2 * n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        out[2*i] = v1[2*i] - v2[2*i];
        out[2*i+1] = v1[2*i+1] - v2[2*i+1];
    }

    return out;
}


void* mulScalarVector( double* v1, double scalar, int n ) {
    double * out = (double *) malloc( n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        out[i] = scalar * v1[i];
    }

    return out;
}

void* mulComplexScalarVector( double* v1, double real, double imaginary, int n ) {
    double * out = (double *) malloc( 2 * n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        out[2*i] = real * v1[i];
        out[2*i+1] = imaginary;
    }

    return out;
}

void* mulComplexScalarComplexVector( double* v1, double real, double imaginary, int n ) {
    double * out = (double *) malloc( 2 * n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        out[2*i] = real * v1[2*i];
        out[2*i+1] = imaginary * v1[2*i+1];
    }

    return out;
}

void* mulFloatScalarComplexVector( double* v1, double real, int n ) {
    double * out = (double *) malloc( 2 * n * sizeof(double) );
    // #pragma omp parallel for
    for (int i=0; i<n; i++) {
        out[2*i] = real * v1[2*i];
        out[2*i+1] = v1[2*i+1];
    }

    return out;
}

void mulScalarMatRow( void* m, double scalar, int nrow, int ncols, int row) {
    //void* v1Dev, void* v2Dev, int n ) {
    
    // return (void *)NULL;
}

void mulScalarMatCol( void* m, double scalar, int nrow, int ncols, int col) {
    //void* v1Dev, void* v2Dev, int n ) {
    
    // return (void *)NULL;
}


void* matVecMul1(  void* mDev, void* vDev, int ncols, int nrows ) {
    //void* v1Dev, void* v2Dev, int n ) {
    
    return (void *)NULL;
}

void* matVecMul2(  void* mDev, void* vDev, int ncols, int nrows ) {
    //void* v1Dev, void* v2Dev, int n ) {
    
    return (void *)NULL;
}

int matrix_get_complex_real_index(int ncol, int i, int j){
    return 2 * (i * ncol + j);
}

int matrix_get_real_index(int ncol, int i, int j){
    return (i * ncol + j);
}

int matrix_get_complex_imag_index(int ncol, int i, int j){
    return matrix_get_complex_real_index(ncol, i, j) + 1;
}

void* matMulFloat(  double* m1, double* m2, int nrows, int ncols, int ncol_m1 ) {
    
    double * out = (double *) malloc( nrows * ncols * sizeof(double) );
    #pragma omp parallel for collapse(2)
    for (int i=0; i<nrows; i++) {
        for(int j=0; j < ncols; j++) {
            double sum = 0;
            double v1;
            double v2;
            #pragma omp unroll
            for(int k=0; k < ncol_m1; k++) {
                v1 = m1[i*ncol_m1+k]; //m1 row
                v2 = m2[k*ncols+j];   //m2 col
                sum += v1 * v2;
            }
            out[i*ncols+j] = sum;
        }    
    }

    return out;
}

void* matMulComplex(  double* m1, double* m2, int nrows, int ncols, int ncol_m1 ) {
    
    double * out = (double *) malloc( 2 * nrows * ncols * sizeof(double) );
    #pragma omp parallel for collapse(2)
    for (int i=0; i<nrows; i++) {
        for(int j=0; j < ncols; j++) {
            int k;
            double sumre = 0, sumim = 0;
            double re1, im1, re2, im2;
            #pragma omp unroll
            for(k=0; k < ncol_m1; k++) {
                int idx = matrix_get_complex_real_index(ncols,i,k);
                re1 = m1[idx];
                im1 = m1[idx+1];

                idx = matrix_get_complex_real_index(ncols,k,j);
                re2 = m2[idx];
                im2 = m2[idx+1];

                sumre += re1*re2 - im1*im2;
                sumim += re1*im2 + re2*im1;
            }
            int idx_out = matrix_get_complex_real_index(ncols,i,j);
            out[idx_out] = sumre; 
            out[idx_out+1] = sumim;
        }    
    }

    return out;
}

void* matMulFloatComplex(  double* m1, double* m2, int nrows, int ncols, int ncol_m1 ) {
    
    double * out = (double *) malloc( 2 * nrows * ncols * sizeof(double) );
    #pragma omp parallel for collapse(2)
    for (int i=0; i<nrows; i++) {
        for(int j=0; j < ncols; j++) {
            int k;
            double sumre = 0, sumim = 0;
            double re1, re2, im2;
            #pragma omp unroll
            for(k=0; k < ncol_m1; k++) {
                int idx = matrix_get_complex_real_index(ncol_m1,i,k);
                re1 = m1[idx];

                idx = matrix_get_complex_real_index(ncols,k,j);
                re2 = m2[idx];
                im2 = m2[idx+1];

                sumre += re1*re2;
                sumim += re1*im2;
            }
            int idx_out = matrix_get_complex_real_index(ncols,i,j);
            out[idx_out] = sumre;
            out[idx_out+1] = sumim;
        }    
    }
    return out;
}

void* matMul(  double* m1, double* m2, int nrows, int ncols, int ncol_m1, int atype, int btype ) {
    
    if( atype == T_FLOAT && btype == T_FLOAT) {
        return matMulFloat(m1, m2, nrows, ncols, ncol_m1);
        
    } else if( atype == T_COMPLEX && btype == T_COMPLEX ) {
        return matMulComplex(m1, m2, nrows, ncols, ncol_m1);
    
    } else if( atype == T_FLOAT && btype == T_COMPLEX ) {
        return matMulFloatComplex(m1, m2, nrows, ncols, ncol_m1);

    }
}

void* matMul2(  void* m1Dev, void* m2Dev, int nrows, int ncols, int qq ) {
    //void* v1Dev, void* v2Dev, int n ) {
    
    return (void *)NULL;
}

void matSquare( void* * outLin, void* * idxOutLin, 
                void* * outCol, void* * idxOutCol, 
                void* mLin, void* idxLin, 
                void* mCol, void* idxCol, 
                int maxcols, int N ) {
    //void* v1Dev, void* v2Dev, int n ) {
    
    // return (void *)NULL;
}

void matVecMul3BD(  double* mat, double* vecIn, double* vecOut, int ncols, int nrows ) {
    #pragma omp parallel for
    for (int i=0; i<nrows; i++) {
        double sum = 0;
        #pragma omp unroll
        for(int j=0; j < ncols; j++) {
            double v1;
            double v2;
            int idx1 = matrix_get_real_index(ncols, i, j);
            v1 = mat[idx1];
            v2 = vecIn[j];
            sum += v1 * v2;
        }
            int idx_out = i; //matrix_get_real_index(ncols, i, j);
            vecOut[idx_out] = sum;
    }
    return ;    
}

void* matVecMul3(  double* mat, double* vecIn, int ncols, int nrows ) {
    double * out = (double *) malloc( nrows * sizeof(double) );
    #pragma omp parallel for
    for (int i=0; i<nrows; i++) {
        double sum = 0;
        #pragma omp unroll
        for(int j=0; j < ncols; j++) {
            double v1;
            double v2;
            int idx1 = matrix_get_real_index(ncols, i, j);
            v1 = mat[idx1];
            v2 = vecIn[j];
            sum += v1 * v2;
        }
            int idx_out = i; //matrix_get_real_index(ncols, i, j);
            out[idx_out] = sum;
    }
    return out;    
}

void computeRowptrU(const smatrix_t* S, const smatrix_t* C, smatrix_t* U) {
    U->row_ptr[0] = 0; // First row starts at index 0
    for (int i = 0; i < S->nrow; i++) {
        int permuted_row = S->col_idx[i];  // Get the row index in C
        int nnz_in_C_row = C->row_ptr[permuted_row + 1] - C->row_ptr[permuted_row];
        U->row_ptr[i + 1] = U->row_ptr[i] + nnz_in_C_row;
    }
}
#include <omp.h>
void computeU(const smatrix_t* S, const smatrix_t* C, smatrix_t* U) {
   printf(" em computeU + Hiagogo, S->type = %d, C->type = %d, U->type = %d\n", S->type, C->type, U->type); //  exit(128+13+7);
    #pragma omp parallel 
{
        int k=0;
    printf("BD em %s: void * void computeU, Thread %d of %d created!\n", __FILE__, omp_get_thread_num(), omp_get_num_threads()); 
    if (C->type == T_COMPLEX ) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < S->nrow; ++row) {
            if (row == 0) { printf("Thread %d of %d created!\n", omp_get_thread_num(), omp_get_num_threads()); }
            int permuted_row = S->col_idx[row]; // Since S is a permutation matrix
            int startC = C->row_ptr[permuted_row], endC   = C->row_ptr[permuted_row + 1];
            int startU = U->row_ptr[row];
            int j = C->col_idx[startC];
            for (int i = 0; i < (endC - startC); i++) {
                U->col_idx[startU + i]          = j++;
                U->values[2 * (startU + i)    ] = C->values[2 * (startC + i)];
                U->values[2 * (startU + i) + 1] = C->values[2 * (startC + i) + 1];
            }
        }
    } else if (C->type == T_FLOAT ) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < S->nrow; ++row) {
            if (k++ < 3 ) {printf("Thread %d of %d created!, row=%d \n",  omp_get_thread_num(), omp_get_num_threads(), row); }
            int permuted_row = S->col_idx[row]; // Since A is a permutation matrix
            int startC = C->row_ptr[permuted_row], endC   = C->row_ptr[permuted_row + 1];
            int startU = U->row_ptr[row];
            //Only for block diagonal matrices
            int j = C->col_idx[startC];
            for (int i = 0; i < (endC - startC); i++) {
                U->col_idx[startU + i] = j++; //C->col_idx[startC + i];
                U->values[startU + i]  = C->values[startC + i];
            }
        }
    } else {
        fprintf(stderr, "Incompatible types in computeU\n"); exit(EXIT_FAILURE);
    }
 } // end pragma omp parallel 
}

 void * permuteSparseMatrix( void * S_, void *C_, void *U_){
      smatrix_t* S = (smatrix_t*) S_;
      smatrix_t* C = (smatrix_t*) C_;
      smatrix_t* U = (smatrix_t*) U_;
      computeRowptrU(S, C, U);
      #pragma omp parallel 
      printf("BD em %s: void * permuteSparseMatrix\n", __FILE__); 
      computeU(S, C, U);
   }


//[Hiago] 
void sparseVecMul(void* vecIn_, void* vecOut_, void* m_values, void* m_row_ptr, void* m_col_idx, int m_nrows, int nnz) {
    printf("BD, em hiperblas-core/src/libhiperblas-cpu-bridge-api-impl.c: void* sparseVecMul(void* v, ...  {\n");
    if (!vecIn_) { fprintf(stderr, "Error: Input vector (vecIn_) is NULL in sparseVecMul.\n"); return ; }
    if (!vecOut_) { fprintf(stderr, "Error: Input vector (vecOut_) is NULL in sparseVecMul.\n"); return ; }
    if (!m_values) { fprintf(stderr, "Error: Matrix values (m_values) is NULL in sparseVecMul.\n"); return ; }
    if (!m_row_ptr) { fprintf(stderr, "Error: Row pointer array (m_row_ptr) is NULL in sparseVecMul.\n"); return ; }
    if (!m_col_idx) { fprintf(stderr, "Error: Column index array (m_col_idx) is NULL in sparseVecMul.\n"); return ; }
    
    double* vec_in  = (double*)vecIn_;
    double* vec_out = (double*)vecOut_;

    long long int* row_ptr   = (long long int*)m_row_ptr;
    long long int* col_idx   = (long long int*)m_col_idx;
    double* values = (double*)m_values;
    #pragma omp parallel for
    for (int row = 0; row < m_nrows; row++) {
        double sum = 0.0;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            int col = col_idx[j];
            double prod = values[j] * vec_in[col];
 //           printf("BD, sum += values[%2d] * vec_in[%2d], %8.4f +=  %8.4f * %8.4f = %8.4f\n", j, col, sum, values[j], vec_in[col], sum+prod);
            sum+=prod;
        }
        vec_out[row] = sum;
//        printf("BD,                vec_out[%2d]          =  %8.4f\n", row, sum);
	    
    }
    //printf("BD, em final de sparseVecMul(void* v, void* m_values, void* m_row_ptr, void* m_col_idx, int m_nrows, int nnz) {\n");
    //  printf("\n%s, linha %d \n", __FILE__, __LINE__ ); exit(128+13);
    return ;
}

void sparseComplexVecMul(void* vecIn_, void* vecOut_, void* m_values, void* m_row_ptr, void* m_col_idx, int m_nrows, int nnz) {
    printf("BD, em hiperblas-core/src/libhiperblas-cpu-bridge-api-impl.c: void* sparseComplexVecMul(void* v, ...  {\n");
    if (!vecIn_) { fprintf(stderr, "Error: Input vector (vecIn_) is NULL in sparseVecMul.\n"); return ; }
    if (!vecOut_) { fprintf(stderr, "Error: Input vector (vecOut_) is NULL in sparseVecMul.\n"); return ; }

    if (!m_values) {
        fprintf(stderr, "Error: Matrix values (m_values) is NULL in sparseVecMul.\n");
        return ;
    }
    if (!m_row_ptr) {
        fprintf(stderr, "Error: Row pointer array (m_row_ptr) is NULL in sparseVecMul.\n");
        return ;
    }
    if (!m_col_idx) {
        fprintf(stderr, "Error: Column index array (m_col_idx) is NULL in sparseVecMul.\n");
        return ;
    }
   /* 
    double* vec_out = (double*)malloc(2 * m_nrows * sizeof(double));
    if (!vec_out) {
        fprintf(stderr, "Error: Memory allocation failed for vec_out.\n");
        return NULL;
    }
    */
    double* vec_in = (double*)vecIn_; // Real and imaginary parts are stored separately
    double* vec_Out = (double*)vecOut_; // Real and imaginary parts are stored separately
    long long int* row_ptr = (long long int*)m_row_ptr;
    long long int* col_idx = (long long int*)m_col_idx;
    double* values = (double*)m_values; // Store real and imaginary parts separately
    
    //printf(" posicoes = "); for (int row = 0; row < m_nrows; row++) printf(" %d, ", row_ptr[row]); printf("\n");
    #pragma omp parallel for
    for (int row = 0; row < m_nrows; row++) {
            int start = row_ptr[row], end = row_ptr[row + 1];
            int     k = col_idx[start];
            double aSum = 0.0, bSum = 0.0;
     //       printf(" row = %d, start = %d, end = %d, k = %d\n", row, start, end, k );
            for (int j = start; j < end; j++) {
                double aS = values[2*j];
                double bS = values[2*j+1];
                double aV = vec_in[2*k];    // Usar col, não k
                double bV = vec_in[2*k +1];  // Usar col, não k


 //               double aV = v->value.f[ 2 * S->col_idx[j] ];    // Usar col, não k
  //              double bV = v->value.f[ 2 * S->col_idx[j] + 1 ];  // Usar col, não k
                double aP = aS * aV - bS * bV;
                double bP = aS * bV + bS * aV;
//            printf("BD, aSum += valuesR[%2d] * vec_inR[%2d], %8.4f +=  %8.4f * %8.4f = %8.4f\n", j, k, aSum, values[2*j], vec_in[2*k], aSum+aP);
 //           printf("BD, bSum += valuesI[%2d] * vec_inI[%2d], %8.4f +=  %8.4f * %8.4f = %8.4f\n", j, k, aSum, values[2*j+1], vec_in[2*k+1], bSum+bP);
                aSum += aP; bSum += bP;
                k++;
            }
            vec_Out[2 * row]     = aSum;
            vec_Out[2 * row + 1] = bSum;
  //      printf("BD,                vec_outR[%2d]          =  %8.4f\n", row, aSum);
   //     printf("BD,                vec_outI[%2d]          =  %8.4f\n", row, bSum);
        }
    //printf("BD, em final de sparseVecMul(void* v, void* m_values, void* m_row_ptr, void* m_col_idx, int m_nrows, int nnz) {\n");
    //  printf("\n%s, linha %d \n", __FILE__, __LINE__ ); exit(128+13);
    // Initialize output vector to zero to avoid undefined behavior
    //exit(128+33);
    //free(vec_in);
    //vec_in=vec_out;
    return; 
}

void* matVecMul3Complex(  double* mat, double* vec, int ncols, int nrows ) {
    
    // printf("matVecMul3Complex 1 ---------\n");
    double * out = (double *) malloc( 2 * nrows * sizeof(double) );
    // printf("matVecMul3Complex 2\n");
    
    #pragma omp parallel for
    for (int i=0; i<nrows; i++) {
        double sumre = 0, sumim = 0;
        for(int j=0; j < ncols; j++) {

            double re1, im1, re2, im2;
            int idx = matrix_get_complex_real_index(ncols,i,j);
            re1 = mat[idx];
            im1 = mat[idx+1];

            idx = 2*j;
            re2 = vec[idx];
            im2 = vec[idx+1];

            sumre += (re1*re2) - (im1*im2);
            sumim += (re1*im2) + (re2*im1);
        }
            int idx_out = 2 * i; //matrix_get_complex_real_index(ncols,i,j);
            out[idx_out] = sumre; 
            out[idx_out+1] = sumim;
    }    

    return out;
}

void* matTranspose(  void* mDev, int ncols, int nrows ) {
//    uint i = get_global_id(0);
//	uint j = get_global_id(1);
//
//	if((i < ncols) && (j < nrows))
//	{
//		uint idx = j * ncols + i;
//		block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = m[idx];
//	}
//
//	barrier(CLK_LOCAL_MEM_FENCE);
//	i = get_group_id(1) * BLOCK_DIM + get_local_id(0);
//	j = get_group_id(0) * BLOCK_DIM + get_local_id(1);
//	if((i < nrows) && (j < ncols))
//    {
//		unsigned int index_out = j * nrows + i;
//		out[index_out] = block[get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1)];
//	}
    
    return (void *)NULL;
}

double sumVector( double* v1, int len ) {

    double out = 0;
    for (int i=0; i<len; i++) {
        out += v1[i];
    }

    return out;
    
}

double normVector( void* vDev, int len ) {
//    unsigned int tid = get_local_id(0);
//    unsigned int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);
//
//    sdata[tid] = (i < n) ? m[i]*m[i] : 0;
//    if ((i + get_local_size(0)) < n) {
//        sdata[tid] += m[i+get_local_size(0)]*m[i+get_local_size(0)];
//    }  
//
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    for(unsigned int s=get_local_size(0)/2; s>0; s>>=1) 
//    {
//        if (tid < s) 
//        {
//            sdata[tid] += sdata[tid + s];
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    if (tid == 0) out[get_group_id(0)] = sdata[0];
    
    return 0.0;
}


double dotVector(void* v1Dev, void* v2Dev, int len ) {
    
    double sum = 0;

    double * v1 = (double *) v1Dev;
    double * v2 = (double *) v2Dev;
    for (int i = 0; i < len; i++) {
        // printf("%d v1=%f v2=%f\n", i, v1[i], v2[i]);
        sum += v1[i] * v2[i];
    }    

    return sum;
}

void dotVectorComplex( double * out_re, double * out_im,  void* v1Dev, void* v2Dev, int len ) {
//    unsigned int tid = get_local_id(0);
//    unsigned int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);
//
//    sdata_re[tid] = (i < n) ? m1[2*i]*m2[2*i] -  m1[2*i+1]*m2[2*i+1]: 0;
//    sdata_im[tid] = (i < n) ? m1[2*i]*m2[2*i+1] + m1[2*i+1]*m2[2*i] : 0;
//    if ((i + get_local_size(0)) < n) {
//        sdata_re[tid] += m1[2*(i+get_local_size(0))]*m2[2*(i+get_local_size(0))] - m1[2*(i+get_local_size(0))+1]*m2[2*(i+get_local_size(0))+1];
//        sdata_im[tid] += m1[2*(i+get_local_size(0))+1]*m2[2*(i+get_local_size(0))] + m1[2*(i+get_local_size(0))]*m2[2*(i+get_local_size(0))+1];
//        
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//    for(unsigned int s=get_local_size(0)/2; s>0; s>>=1)
//    {
//        if (tid < s)
//        {
//            sdata_im[tid] += sdata_im[tid + s];
//            sdata_re[tid] += sdata_re[tid + s];
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    if (tid == 0){ out_re[get_group_id(0)] = sdata_re[0];out_im[get_group_id(0)] = sdata_im[0]; }
    
    // return (void *)NULL;
}

    
#endif
