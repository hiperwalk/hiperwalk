#ifndef __LIBNEBLINA_H_
#define __LIBNEBLINA_H_

#ifdef	__cplusplus
extern "C" {
#endif
    

#include "hiperblas.h"
#include "hiperblas_vector.h"
#include "hiperblas_matrix.h"
#include "hiperblas_smatrix.h"
#include "hiperblas_complex.h"
#include "bridge_api.h"
//#include "clutils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ivget(vet,i) ((vet).value.i[i-1])
#define fvget(vet,i) ((vet).value.i[i-1])
#define COMPLEX_SIZE (2 * sizeof(double))


 void ** movetodev      ( void ** i, int * status );
 void ** movetohost     ( void ** i, int * status );

void ord_smat( double * m, int * idx, int max, int N );
void smatrix_line_to_col( double * out, int * idx_out, double * in, int * idx_in, int max, int N );
void print_data_type( data_type t );

void load_plugin(bridge_manager_t *m, char* library_name, int idx);
void release_plugin(bridge_manager_t *m, int idx);

void ** hiperblas_sparse( void ** i, int * status );

matrix_t * matrix_multiply( matrix_t * a, matrix_t * b );
vector_t * smatvec_multiply( bridge_manager_t *m, int index, smatrix_t * a, vector_t * b );
vector_t * matvec_multiply( matrix_t * a, vector_t * b );

void clear_input( void ** i, int nparams );

void hiperblas_strtype( data_type type, char out[256] );


#ifdef	__cplusplus
}
#endif

#endif




