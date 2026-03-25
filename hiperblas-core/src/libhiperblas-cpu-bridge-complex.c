#include "libhiperblas.h"
#include "hiperblas_complex.h"
#include <stdio.h>
#include <stdlib.h>

complex_t * complex_new( double real, double imaginary ) {
    complex_t * ret = (complex_t *) malloc( sizeof( complex_t ) );

    ret->re = real;
    ret->im = imaginary;
    
    return ret;
}

void complex_delete( complex_t * c ) {
    
    free (c);
}

