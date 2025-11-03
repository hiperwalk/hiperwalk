/* 
 * File:   hiperblas_vector.h
 * Author: paulo
 *
 * Created on 24 de agosto de 2021, 23:09
 */

#ifndef NEBLINA_VECTOR_H
#define NEBLINA_VECTOR_H

#ifdef __cplusplus
extern "C" {
#endif
#include "hiperblas.h"

typedef struct __vector_t {
    data_vector_u      value;
    int                len;
    data_type          type;
    unsigned char      location;
    void*              extra;
    int                externalData;
} vector_t;

//vector_t * vector_new( int len, data_type type ) ;
//void vector_delete( vector_t * v ) ;
//void vecreqhost( vector_t * v );
//void vecreqdev ( vector_t * v );


#ifdef __cplusplus
}
#endif

#endif /* NEBLINA_VECTOR_H */

