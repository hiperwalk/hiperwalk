/* 
 * File:   hiperblas_matrix.h
 * Author: paulo
 *
 * Created on 25 de agosto de 2021, 00:07
 */

#ifndef HIPERBLAS_MATRIX_H
#define HIPERBLAS_MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif
#include "hiperblas.h"
#include "hiperblas_complex.h"
#include <stdio.h>
#include <stdlib.h>
    
 typedef struct __matrix_t {
    data_vector_u      value;
    int                ncol;
    int                nrow;
    data_type          type;
    unsigned char    location;
    void*             extra;
    int               externalData;
} matrix_t;

//matrix_t * matrix_new( int nrow, int ncol, data_type type );
//void matrix_delete( matrix_t * v );
//void matreqhost( matrix_t * v );
//void matreqdev ( matrix_t * v );
//void matrix_set_real_value(matrix_t *  m, int i, int j, double r);
//double matrix_get_real_value(matrix_t *  m, int i, int j);
//void matrix_set_complex_value(matrix_t *  m, int i, int j, double r, double im);
//double matrix_get_complex_real_value(matrix_t *  m, int i, int j);
//double matrix_get_complex_imaginary_value(matrix_t *  m, int i, int j);



#ifdef __cplusplus
}
#endif

#endif /* HIPERBLAS_MATRIX_H */

