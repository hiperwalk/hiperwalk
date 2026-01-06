/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   hiperblas_smatrix.h
 * Author: paulo
 *
 * Created on 26 de agosto de 2021, 00:05
 */

#ifndef HIPERBLAS_SMATRIX_H
#define HIPERBLAS_SMATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __smatrix_t {
    /* old semi-csr structure
    int nrow; int ncol; int maxcols;
    double * m;
    int   * idx_col; int * rcount; int * icount; */

    int nrow;
    int ncol;
    int nnz;

    long int  *row_ptr, *col_idx;
    double* values;

    slist ** smat;
    int isPacked;
            
    data_type       type;
    unsigned char   location;
    void*           extra;
    void*           idxColMem;
    
} smatrix_t;

//smatrix_t * smatrix_new( int nrow, int ncol, data_type type );
//void smatrix_t_clear( smatrix_t * m );
//void smatrix_load_double( smatrix_t * m, FILE * f );
//void smatrix_set_real_value(smatrix_t *  m, int i, int j, double r);
//void smatrix_pack(smatrix_t * m);
//void smatrix_set_complex_value(smatrix_t *  m, int i, int j, double r, double im);
//void smatrix_pack_complex(smatrix_t * m);
//
//void smatrix_load_complex( smatrix_t * m, FILE * f );
//void smatrix_delete( smatrix_t * v );
//void smatreqhost( smatrix_t * v ) ;
//void smatreqdev ( smatrix_t * v );
//
//slist * slist_add( slist * l, int col, double re, double im );
//void slist_clear( slist * l );

#ifdef __cplusplus
}
#endif

#endif /* HIPERBLAS_SMATRIX_H */

