/* 
 * File:   hiperblas_complex.h
 * Author: paulo
 *
 * Created on 16 de outubro de 2021, 11:30
 */

#ifndef HIPERBLAS_COMPLEX_H
#define HIPERBLAS_COMPLEX_H

#ifdef __cplusplus
extern "C" {
#endif
#include "hiperblas.h"

typedef struct __complex_t {
    double re;
    double im;
} complex_t;


//complex_t * complex_new( double real, double imaginary ) ;
//void complex_delete( complex_t * c ) ;


#ifdef __cplusplus
}
#endif

#endif /* HIPERBLAS_COMPLEX_H */

