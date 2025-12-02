#include "hiperblas_std.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>

#include <omp.h>

#include "libhiperblas.h"
#include "hiperblas.h"
#include "hiperblas_list.h"
#include "bridge_api.h"


void runerror( char * strerr ) {
    fprintf(stderr, " runtime error: %s\n", strerr);
    exit( 1 );
}

// int vec_len(bridge_manager_t *m, int index,  void ** i, int * status ) {
//     object_t out;
//     object_t ** in = (object_t **) i;
//     int len = 0;
//     if( type( *in[0] ) == T_VECTOR ) {
//         vector_t * vec = (vector_t *)vvalue( *in[0] );
//         len = vec->len;  
//     } else if( type( *in[0] ) == T_STRING ) {
//         len = strlen( svalue( *in[0] ) );
//     } else if( type( *in[0] ) == T_LIST ) {
//         len = m->bridges[index].list_len(  (list_t *) vvalue( *in[0] ) );
//     } else {
//         runerror("invalid type input for function len()");      
//     }

//     return len;
// }


 void ** mat_len_col( void ** i, int * status ) {
    object_t  out;// = (object_t *) malloc( sizeof( object_t ) );
    object_t ** in = (object_t **) i;
    type( out ) = T_INT;
    matrix_t * mat = (matrix_t *)vvalue( *in[0] );
    ivalue( out ) = mat->ncol;  
    static void * ret[1];
    clear_input( i, 1 );
    ret[0] = (void *) &out;
    if (status != NULL) {
        *status = 0;
    }
    return ret;
}


 void ** mat_len_row( void ** i, int * status ) {
    object_t ** in = (object_t **) i;
    if (status != NULL) {
        *status = 0;
    }
    if( type( *in[0] ) == T_MATRIX ) {
        object_t out;// = (object_t *) malloc( sizeof( object_t ) );

        type( out ) = T_INT;
        matrix_t * mat = (matrix_t *)vvalue( *in[0] );
        ivalue( out ) = mat->nrow;  
        static void * ret[1];
        clear_input( i, 1 );
        ret[0] = (void *) &out;
        return ret;
    } else {
        if (status != NULL) {
            *status = -1;
        }
        runerror( "Runtime error: no matrix found\n");
    }   
    return NULL;    
}
//
// void ** mat_mul_cpu( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        matrix_t * a = (matrix_t *) vvalue( *in[1] );
//        matrix_t * b = (matrix_t *) vvalue( *in[0] );
//        matreqhost( a ); matreqhost( b ); 
//        object_t out;// = (object_t *) malloc( sizeof( object_t ) );
//        matrix_t * r = matrix_multiply( a, b );
//        r->location = LOCHOS;
//        type( out ) = T_MATRIX;
//        vvalue( out ) = (void *) r;
//        static void * ret[1];
//        clear_input( i, 2 );
//        ret[0] = (void *) &out;
//        return ret;
//}
// 
void delete_object_array(object_t ** in, int len){
    if (in != NULL) {
        for (int i =0; i < len; i ++){
            delete_object(in[i]);
        }
        free(in);
    }
}

void delete_object(object_t * in){
    if (in != NULL) {
        free( in );
    }
}

object_t ** convertToObject(vector_t * a, vector_t * b) {
    object_t ** in;
     if (b != NULL) {
        in = (object_t **) malloc(2 * sizeof(object_t *));
        in[1] = (object_t *) malloc(sizeof(object_t));
        // vvalue( *in[1] ) = b; 
        in[1]->value.v = (void *)b;
        in[1]->type = T_VECTOR;
     } else {
        in = (object_t **) malloc(2 * sizeof(object_t *));
     }
    
    in[0] = (object_t *) malloc(sizeof(object_t));
    //vvalue( *in[0] ) = a; 
    in[0]->value.v = (void *)a;
    in[0]->type = T_VECTOR;
    return in;
 }

object_t ** convertToObject3BD(matrix_t * m, vector_t * a, vector_t * b) {
    object_t ** in;
    in = (object_t **) malloc(3 * sizeof(object_t *));

    in[0] = (object_t *) malloc(sizeof(object_t));
    vvalue( *in[0] ) = m; in[0]->type = T_MATRIX;

    in[1] = (object_t *) malloc(sizeof(object_t));
    vvalue( *in[1] ) = a; in[1]->type = T_VECTOR;

    in[2] = (object_t *) malloc(sizeof(object_t));
    vvalue( *in[2] ) = b; in[2]->type = T_VECTOR;
    return in;
 }

object_t ** convertToObject3(vector_t * a, matrix_t * b) {
    object_t ** in;
     if (b != NULL) {
        in = (object_t **) malloc(2 * sizeof(object_t *));
        in[1] = (object_t *) malloc(sizeof(object_t));
        vvalue( *in[1] ) = b; in[1]->type = T_MATRIX;
     } else {
        in = (object_t **) malloc(sizeof(object_t *));
     }
    
    in[0] = (object_t *) malloc(sizeof(object_t));
    vvalue( *in[0] ) = a; 
    in[0]->type = T_VECTOR;
    
    return in;
 }

object_t ** convertMatMatToObject(matrix_t * a, matrix_t * b) {
    object_t ** in;
    in = (object_t **) malloc(2 * sizeof(object_t *));
    
    in[0] = (object_t *) malloc(sizeof(object_t));
    vvalue( *in[0] ) = a; 
    in[0]->type = T_MATRIX;
    
    in[1] = (object_t *) malloc(sizeof(object_t));
    vvalue( *in[1] ) = b; 
    in[1]->type = T_MATRIX;
    
    return in;
 }

object_t ** convertScaVecToObject(double s, vector_t * a) {
    object_t ** in;
    in = (object_t **) malloc(2 * sizeof(object_t *));
    
    in[0] = (object_t *) malloc(sizeof(object_t));
    fvalue( *in[0] ) = s; 
    in[0]->type = T_FLOAT;
    
    in[1] = (object_t *) malloc(sizeof(object_t));
    vvalue( *in[1] ) = a; 
    in[1]->type = T_VECTOR;
    
    return in;
 }

object_t ** convertScaMatToObject(double s, matrix_t * a) {
    object_t ** in;
    in = (object_t **) malloc(2 * sizeof(object_t *));
    
    in[0] = (object_t *) malloc(sizeof(object_t));
    fvalue( *in[0] ) = s; 
    in[0]->type = T_FLOAT;
    
    in[1] = (object_t *) malloc(sizeof(object_t));
    vvalue( *in[1] ) = a; 
    in[1]->type = T_MATRIX;
    
    return in;
 }

object_t ** BD00convertToObject4(vector_t * a, smatrix_t * b) {
    object_t ** in;
     if (b != NULL) {
        in = (object_t **) malloc(2 * sizeof(object_t *));
        in[1] = (object_t *) malloc(sizeof(object_t));
        vvalue( *in[1] ) = b; 
        in[1]->type = T_SMATRIX;
     } else {
        in = (object_t **) malloc(sizeof(object_t *));
     }
    in[0] = (object_t *) malloc(sizeof(object_t));
    vvalue( *in[0] ) = a; 
    in[0]->type = T_VECTOR;
    return in;
 }

object_t ** convertToObject4BD(smatrix_t * m, vector_t * vI, vector_t * vO){
	//by bidu@lncc.br in nov.2025
    object_t ** in = (object_t **) malloc(3 * sizeof(object_t *));
    in[0] = (object_t *) malloc(sizeof(object_t));
    in[0]->type = T_SMATRIX;
    vvalue( *in[0] ) = m; 
    in[1] = (object_t *) malloc(sizeof(object_t));
    in[1]->type = T_VECTOR;
    vvalue( *in[1] ) = vI; 
    in[2] = (object_t *) malloc(sizeof(object_t));
    in[2]->type = T_VECTOR;
    vvalue( *in[2] ) = vO; 
    return in;
 }
object_t ** convertToObject4(vector_t * a, smatrix_t * b) {
	//by bidu@lncc.br in set.2025
    object_t ** in = (object_t **) malloc(2 * sizeof(object_t *));
    in[0] = (object_t *) malloc(sizeof(object_t));
    in[0]->type = T_VECTOR;
    vvalue( *in[0] ) = a; 
    in[1] = (object_t *) malloc(sizeof(object_t));
    in[1]->type = T_SMATRIX;
    vvalue( *in[1] ) = b; 
    return in;
 }

void ** copy_vector_from_device( bridge_manager_t *m, int idx, void ** i, int * status ) {
        
        // a pre condition is that data should be on the device, if not should we throw an erro?

        object_t ** in = (object_t **) i;
        vector_t * a = (vector_t *) vvalue( *in[0] );
        vector_t * r = m->bridges[idx].vector_new(a->len, a->type, 1, NULL);
        // m->bridges[idx].vecreqdev( r ); 
        
        int size = (a->type == T_FLOAT) ? sizeof(double) : (sizeof(double) * 2); 

        r->value.f = (void*)m->bridges[idx].copyVectorFromDevice_f( a->extra, (size * a->len) ); 

        // m->bridges[idx].vecreqhost( r ); 

        if (status != NULL) {
            *status = 0;
        }
        return (void *) r;
}

 void ** vec_add( bridge_manager_t *m, int index, void ** i, int * status ) {
        
        object_t ** in = (object_t **) i;
        vector_t * a = (vector_t *) vvalue( *in[0] );
        vector_t * b = (vector_t *) vvalue( *in[1] );
        vector_t * r = m->bridges[index].vector_new(b->len, b->type, 0, NULL);
        //apenas para CPU
        // free(r->value.f);
        m->bridges[index].vecreqdev( a ); 
        m->bridges[index].vecreqdev( b ); 
        m->bridges[index].vecreqdev( r ); 
        
        if (b->type == T_FLOAT) {
            r->extra = (void*)m->bridges[index].addVectorF_f( a->extra, b->extra, b->len ); 
            
        } else if (b->type == T_COMPLEX) {
            r->extra = (void*)m->bridges[index].addVectorC_f( a->extra, b->extra, b->len ); 
        }

        if (status != NULL) {
            *status = 0;
        }
        return (void *) r;
}
//
 void ** vec_conj( bridge_manager_t *m, int index, void ** i, int * status ) {
        object_t ** in = (object_t **) i;
        vector_t * a = (vector_t *) vvalue( *in[0] );
        vector_t * r = m->bridges[index].vector_new(a->len, T_COMPLEX, 0, NULL);
        //apenas para cpu
        // free( r->value.f);
        m->bridges[index].vecreqdev( a ); m->bridges[index].vecreqdev( r );

        r->extra = (void*)m->bridges[index].vecConjugate_f( a->extra, a->len ); 
    
        clear_input( i, 1 );
        if (status != NULL) {
            *status = 0;
        }
        return (void *) r;
}
// void ** vec_conjugate( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        vector_t * a = (vector_t *) vvalue( *in[0] );
//        object_t out;
//        vector_t * r = (vector_t *) malloc( sizeof( vector_t ) );
//        vecreqdev( a );
//
//        r->extra = (void*)vecConjugate( a->extra, a->len ); 
//    
//        r->len = a->len;
//        r->type = T_COMPLEX;
//        r->location = LOCDEV;
//        r->value.f = NULL;
//        type( out ) = T_VECTOR;
//        
//        vvalue( out ) = (void *) r;
//        static void * ret[1];
//        ret[0] = (void *) &out;
//        clear_input( i, 1 );
//        return ret;
//}
 void ** vec_prod( bridge_manager_t *m, int index, void ** i, int * status ) {
        object_t ** in = (object_t **) i;
        vector_t * a = (vector_t *) vvalue( *in[0] );
        vector_t * b = (vector_t *) vvalue( *in[1] );
        m->bridges[index].vecreqdev( a ); 
        m->bridges[index].vecreqdev( b );
        vector_t * r = (vector_t *) malloc( sizeof( vector_t ) );
        if( a->type == T_FLOAT ) {
            r->extra = (void*)m->bridges[index].prodVector_f( a->extra, b->extra, b->len ); 
        } else {
            r->extra = (void*)m->bridges[index].prodComplexVector_f( a->extra, b->extra, b->len ); 
        }

        r->len = b->len;
        r->type = a->type;
        r->location = LOCDEV;
        r->value.f = NULL;
        clear_input( i, 2 );
        if (status != NULL) {
            *status = 0;
        }
        return (void *) r;

}

 void ** vec_sum( bridge_manager_t *m, int index, void ** i, int * status ) {
        object_t ** in = (object_t **) i;
        vector_t * a = (vector_t *) vvalue( *in[0] );
        object_t * out = (object_t *) malloc(sizeof(object_t));
        
        m->bridges[index].vecreqdev( a );
        out->value.f = m->bridges[index].sumVector_f( a->extra, a->len );
        out->type  = T_FLOAT;

        if (status != NULL) {
            *status = 0;
        }
        return (void *) out;
}

 void ** vec_norm( bridge_manager_t *m, int index, void ** i, int * status ) {
        object_t ** in = (object_t **) i;
        vector_t * a = (vector_t *) vvalue( *in[0] );
        object_t * out = (object_t *) malloc( sizeof( object_t ) );
        m->bridges[index].vecreqdev( a );
        fvalue( *out ) = m->bridges[index].normVector_f( a->extra, a->len ); 
        type( *out ) = T_FLOAT;
        static void * ret[1];
        ret[0] = (void *) out;
        if (status != NULL) {
            *status = 0;
        }
        return ret;
}
 void ** vec_dot( bridge_manager_t *m, int index, void ** i, int * status ) {
        object_t ** in = (object_t **) i;
        vector_t * v1 = (vector_t *) vvalue( *in[0] );
        vector_t * v2 = (vector_t *) vvalue( *in[1] );
        m->bridges[index].vecreqdev( v1 ); 
        m->bridges[index].vecreqdev( v2 );
        object_t * out;
        if( type( *v1 ) == T_FLOAT && type( *v2 ) == T_FLOAT ) {
            out = (object_t *) malloc( sizeof( object_t ) );
            fvalue( *out ) = m->bridges[index].dotVector_f(v1->extra, v2->extra, v1->len); 
            type( *out ) = T_FLOAT;
        } else {
            out = (object_t *) malloc( sizeof( object_t ) );
            double re, im;
            m->bridges[index].dotVectorComplex_f(&re, &im, v1->extra, v2->extra, v1->len); 
            complex_t * res = (complex_t *) malloc( sizeof(complex_t) );
            res->im = im;
            res->re = re;
            vvalue( *out ) = (void *) res; 
            type( *out ) = T_COMPLEX;
         }
        
        static void * ret[1];
        ret[0] = (void *) out;
        if (status != NULL) {
            *status = 0;
        }
        return ret;
}
//
// void ** vec_dot_cpu( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        vector_t * v1 = (vector_t *) vvalue( *in[0] );
//        vector_t * v2 = (vector_t *) vvalue( *in[1] );
//        vecreqhost( v1 );vecreqhost( v2 ); 
//        object_t * out = (object_t *) malloc( sizeof( object_t ) );
//        double sum = 0.0;
//        int k = 0;
//        for(k = 0; k < v1->len; k++ )
//            sum += v1->value.f[k]*v2->value.f[k];
//        
//        fvalue( *out ) = sum; 
//        type( *out ) = T_FLOAT;
//        static void * ret[1];
//        ret[0] = (void *) out;
//        return ret;
//}
//
// void ** vec_norm_cpu( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        vector_t * a = (vector_t *) vvalue( *in[0] );
//        object_t * out = (object_t *) malloc( sizeof( object_t ) );
//        vecreqhost( a );
//        double sum = 0.0;
//        int k = 0;
//        for(k=0;k<a->len;k++) {
//            sum += a->value.f[k]*a->value.f[k];
//        }
//        fvalue( *out ) = sqrt( sum ); 
//        type( *out ) = T_FLOAT;
//        static void * ret[1];
//        ret[0] = (void *) out;
//        return ret;
//}
// void ** vec_sum_cpu( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        vector_t * a = (vector_t *) vvalue( *in[0] );
//        object_t * out = (object_t *) malloc( sizeof( object_t ) );
//        vecreqhost( a );
//        double sum = 0.0;
//        int k = 0;
//        for(k=0;k<a->len;k++) {
//            sum += a->value.f[k];
//        }
//        fvalue( *out ) = sum; 
//        type( *out ) = T_FLOAT;
//        static void * ret[1];
//        ret[0] = (void *) out;
//        return ret;
//}
//
 void ** vec_sub( bridge_manager_t *m, int index, void ** i, int * status ) {
        object_t ** in = (object_t **) i;
        vector_t * a = (vector_t *) vvalue( *in[0] );
        vector_t * b = (vector_t *) vvalue( *in[1] );
        vector_t * r = m->bridges[index].vector_new(b->len, b->type, 0, NULL);
        // apenas para cpu
        // free(r->value.f);
        m->bridges[index].vecreqdev( a ); 
        m->bridges[index].vecreqdev( b ); 
        m->bridges[index].vecreqdev( r );
        
        if (b->type == T_FLOAT) {
            r->extra = (void*)m->bridges[index].subVector_f( a->extra, b->extra, b->len );
        } else if (b->type == T_COMPLEX) {
            r->extra = (void*)m->bridges[index].subVectorC_f( a->extra, b->extra, b->len ); 
        }
        if (status != NULL) {
            *status = 0;
        }
        return (void *) r;
}
//
// void ** vec_add_cpu( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        vector_t * a = (vector_t *) vvalue( *in[0] );
//        vector_t * b = (vector_t *) vvalue( *in[1] );
//        vecreqhost( a );vecreqhost( b );
//        object_t out;
//        vector_t * r = (vector_t *) malloc( sizeof( vector_t ) );
//        int k;
//        if( a->len != b->len )
//            runerror( "invalid size of vectors" );
//       
//        
//        if( a->type == T_FLOAT && b->type == T_FLOAT ) {
//            r->value.f = (double *) malloc(b->len * sizeof(double));
//            for(k=0;k<b->len;k++)
//                r->value.f[k] = a->value.f[k] + b->value.f[k];
//            r->type = T_FLOAT;
//        } else if( a->type == T_COMPLEX && b->type == T_COMPLEX ) {
//            r->value.f = (double *) malloc(2 * b->len * sizeof(double));
//            for(k=0;k<2*b->len;k++)
//                r->value.f[k] = a->value.f[k] + b->value.f[k];
//            r->type = T_COMPLEX;
//        } else if( (a->type == T_FLOAT && b->type == T_COMPLEX) || (a->type == T_COMPLEX && b->type == T_FLOAT) ) {
//            r->value.f = (double *) malloc(2*b->len * sizeof(double));
//            if( b->type == T_COMPLEX ) {
//                vector_t * tmp = b;
//                b = a;
//                a = tmp;
//            }
//            
//            for(k=0;k<b->len;k++) {
//                r->value.f[2*k] = a->value.f[2*k] + b->value.f[k];
//                r->value.f[2*k+1] = a->value.f[2*k+1];
//            }
//            r->type = T_COMPLEX;
//        }             
//        
//        
//        
//        r->location = LOCHOS;
//        r->len = b->len;
//        type( out ) = T_VECTOR;
//        vvalue( out ) = (void *) r;
//        static void * ret[1];
//        ret[0] = (void *) &out;
//        return ret;
//}
//
// void ** vec_sub_cpu( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        vector_t * a = (vector_t *) vvalue( *in[0] );
//        vector_t * b = (vector_t *) vvalue( *in[1] );
//        vecreqhost( a );vecreqhost( b );
//        object_t * out = (object_t *) malloc( sizeof( object_t ) );
//        vector_t * r = (vector_t *) malloc( sizeof( vector_t ) );
//        r->value.f = (double *) malloc(b->len * sizeof(double));
//        int k;
//        for(k=0;k<b->len;k++)
//            r->value.f[k] = a->value.f[k] - b->value.f[k];           
//        r->location = LOCHOS;
//        r->type = T_FLOAT;
//        r->len = b->len;
//        type( *out ) = T_VECTOR;
//        vvalue( *out ) = (void *) r;
//        static void * ret[1];
//        ret[0] = (void *) out;
//        return ret;
//}

 void ** mat_add( bridge_manager_t *m, int index, void ** i, int * status ) {
    object_t ** in = (object_t **) i;
    matrix_t * a = (matrix_t *) vvalue( *in[0] );
    matrix_t * b = (matrix_t *) vvalue( *in[1] );
    m->bridges[index].matreqdev( a );
    m->bridges[index].matreqdev( b );
    // object_t out;
    matrix_t * r = NULL;

    if( a->type == T_FLOAT && b->type == T_FLOAT ) { 
        r = m->bridges[index].matrix_new(b->ncol,b->nrow,T_FLOAT, 0, NULL);
        r->extra = m->bridges[index].addVectorF_f( a->extra, b->extra, b->nrow * b->ncol );
        r->location = LOCDEV;
    }else if ( a->type == T_COMPLEX && b->type == T_COMPLEX) {
        r = m->bridges[index].matrix_new(b->ncol,b->nrow,T_COMPLEX, 0, NULL);
        r->extra = m->bridges[index].addVectorC_f( a->extra, b->extra, b->nrow * b->ncol );
        r->location = LOCDEV;
    } else if((a->type == T_FLOAT && b->type == T_COMPLEX) ||
              (a->type == T_COMPLEX && b->type == T_FLOAT)) {
        r = m->bridges[index].matrix_new(b->ncol,b->nrow,T_COMPLEX, 0, NULL);
        r->ncol = b->ncol;
        r->nrow = b->nrow;
        r->type = T_COMPLEX;
        if( a->type == T_FLOAT )
            r->extra = m->bridges[index].addVectorFC_f( a->extra, b->extra, b->nrow * b->ncol );
        else
            r->extra = m->bridges[index].addVectorFC_f( b->extra, a->extra, b->nrow * b->ncol );
        r->location = LOCDEV;
        r->value.f = NULL;
        // type( out ) = T_MATRIX;
        // vvalue( out ) = (void *) r;
    }
    if (status != NULL) { *status = 0; }
    return (void *) r;
}
 
 void ** mat_sub( bridge_manager_t *m, int index, void ** i, int * status ) {
    object_t ** in = (object_t **) i;
    matrix_t * a = (matrix_t *) vvalue( *in[1] );
    matrix_t * b = (matrix_t *) vvalue( *in[0] );
    m->bridges[index].matreqdev( a );
    m->bridges[index].matreqdev( b );
    object_t out;// = (object_t *) malloc( sizeof( object_t ) );
    matrix_t * r = (matrix_t *) malloc( sizeof(matrix_t) );
    r->ncol = a->ncol;
    r->nrow = a->nrow;
    r->type = T_FLOAT;
    r->extra = m->bridges[index].subVector_f( a->extra, b->extra, b->nrow * b->ncol );
    r->location = LOCDEV;
    r->value.f = NULL;
    type( out ) = T_MATRIX;
    vvalue( out ) = (void *) r;
    static void * ret[1];
    ret[0] = (void *) &out;
    clear_input(i, 2);
    if (status != NULL) {
        *status = 0;
    }
    return ret;
}
 void ** mat_mul( bridge_manager_t *m, int index, void ** i, int * status ) {
    object_t ** in = (object_t **) i;
    matrix_t * a = (matrix_t *) vvalue( *in[0] );
    matrix_t * b = (matrix_t *) vvalue( *in[1] );
    
    matrix_t * r = NULL;
    long matrix_size = 0;
    if( a->type == T_FLOAT && b->type == T_FLOAT ) {
        r = m->bridges[index].matrix_new(b->ncol,a->nrow,T_FLOAT, 0, NULL);
        r->type = T_FLOAT; 
        matrix_size = b->ncol;
    } else if( (a->type == T_COMPLEX && b->type == T_COMPLEX) || 
             (a->type == T_FLOAT && b->type == T_COMPLEX) ) {
        r = m->bridges[index].matrix_new(b->ncol,a->nrow,T_COMPLEX, 0, NULL);
        r->type = T_COMPLEX;
        matrix_size = 2 * b->ncol;
    } else {
        runerror( "Invalid types for mat_mul\n" );
    }
    struct timeval stop, start, ini, end, tval_result;
    long max_mem = m->bridges[index].get_Engine_Max_Memory_Allocation_f();
    //printf("matrix_size=%ld max_mem=%ld (max_mem / sizeof(double))=%ld\n", matrix_size, max_mem, (max_mem / sizeof(double)));
    //printf("matrix_size < (max_mem / sizeof(double))=%d\n",matrix_size < (max_mem / sizeof(double)));
    if ( 1 ) { //|| (matrix_size * matrix_size) < (max_mem / sizeof(double))
        // gettimeofday(&ini, NULL);
        m->bridges[index].matreqdev( a );
        m->bridges[index].matreqdev( b );
        
    r->extra = (void *)m->bridges[index].matMul_f( a->extra, b->extra, a->nrow, b->ncol, a->ncol, a->type, b->type );
        // gettimeofday(&end, NULL);
            // timersub(&end, &ini, &tval_result);
            // printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    r->location = LOCDEV;
    r->value.f = NULL;
    } else {
        // A * B -> row from A and col from B
        // use max_mem
        printf("processing chunks\n");
        r->location = LOCHOS;
        r->value.f = (double *) calloc( a->nrow * b->ncol, sizeof( double ) );
        printf("callocated\n");

        printf("1\n");
        printf("max_mem=%ld\n", max_mem);
        printf("(max_mem / sizeof(double))=%ld\n", (max_mem / sizeof(double)));
        long qty_chunks = ceil((matrix_size * 1.0) / (max_mem / sizeof(double)));
        printf("qty_chunks=%ld\n", qty_chunks);
        printf("2\n");
        long chunk_size = matrix_size / qty_chunks;
        printf("3\n");
        printf("chunk_size=%ld\n", chunk_size);
        
        

        for (int j = 0; j < b->ncol; j++){ //we will move through columns first
            gettimeofday(&ini, NULL);
            for (int c = 0; c < qty_chunks; c++){ //then we move through the chunks 
                // gettimeofday(&start, NULL);
                double * B_col = m->bridges[index].matrix_copy_col(b, j, c * chunk_size, chunk_size);
                // gettimeofday(&stop, NULL);
                // printf(" col copy took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
                // getchar();
                //printf("B_col=%p\n",B_col);
                // for (int n=0; n <chunk_size; n++){
                //     printf("B_col[%d]=%lf\n", n, B_col[n]);
                // }
                vector_t * col = m->bridges[index].vector_new(chunk_size, T_FLOAT, 0, B_col);
                // for (int n=0; n <col->len; n++){
                //     printf("col[%d]=%lf\n", n, col->value.f[n]);
                //     printf("B_col[%d]=%lf\n", n, B_col[n]);
                // }
                m->bridges[index].vecreqdev( col );
                for(int i = 0; i < a->nrow; i++){
                    //for the same column chunk that was copied to device memory
                    //we calculate the dot product for all the row chunks to leverage
                    //the column that was copied first (columns will inccur in more
                    //cache misses on the CPU)

                    // gettimeofday(&start, NULL);
                    double * A_row = m->bridges[index].matrix_copy_row(a, i, c * chunk_size, chunk_size);
                    // gettimeofday(&stop, NULL);
                    // printf("  row copy took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec); 
                    //printf("A_row=%p\n",A_row);
                    vector_t * row = m->bridges[index].vector_new(chunk_size, T_FLOAT, 0, A_row);
                    // for (int n=0; n <row->len; n++){
                    //     printf("row[%d]=%lf\n", n, row->value.f[n]);
                    //     printf("A_row[%d]=%lf\n", n, A_row[n]);
                    // }
                    // gettimeofday(&start, NULL);
                    m->bridges[index].vecreqdev( row );
                    // gettimeofday(&stop, NULL);
                    // printf("  row move took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec); 
                    //printf("dotVector\n");
                    // gettimeofday(&start, NULL);
                    double res = m->bridges[index].dotVector_f(row->extra, col->extra, chunk_size);
                    // gettimeofday(&stop, NULL);
                    // printf("   dot product took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec); 
                    //printf("%d %d %d res=%f\n", j, c, i, res);
                    r->value.f[i * r->ncol + j] += res;
                    m->bridges[index].vector_delete(row);
                    free(A_row);
                }
                m->bridges[index].vector_delete(col);
                free(B_col);
            }
            printf("j=%d ", j);
            gettimeofday(&end, NULL);
            timersub(&end, &ini, &tval_result);
            printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
            //printf("col took %lu us\n\n", (end.tv_sec - ini.tv_sec) * 1000000 + end.tv_usec - ini.tv_usec);
            //getchar();
        }

    }
    
    //clear_input(i, 2);
    if (status != NULL) {
        *status = 0;
    }
    return (void *)r;
}

// void ** hiperblas_mat_sqr( void ** i, int * s ) {
////    object_t ** in = (object_t **) i;
////    smatrix_t * a = (smatrix_t *) vvalue( *in[0] );
////    double * aCol = (double *) malloc( a->maxcols * 2 * a->nrow * sizeof(double ) );
////    int * aIdxCol = (int *) malloc( a->maxcols * a->nrow * sizeof(int) );
////    cl_int status;
////    
////    smatrix_line_to_col( aCol, aIdxCol, a->m, a->idx_col, a->maxcols, a->nrow );
////    int N = a->ncol;
////    
////    
////    int maxcols =  a->maxcols;
////    ord_smat( aCol, aIdxCol, maxcols, N );
////    ord_smat( a->m, a->idx_col, maxcols, N );
////    /*
////    int j, jj, max = maxcols;
////    for(jj=0; jj < N; jj++ ){
////        for(j=0; j < max && aIdxCol[jj*max+j] != -1; j++ ) {
////            printf("---> (%d, %d) [%f %f]\n", aIdxCol[jj*max+j], jj, aCol[2*(jj*max+j)], aCol[2*(jj*max+j)+1] ); 
////        }
////    }*/
////    int * ptr1 = (int *)malloc(N * sizeof(int)), * ptr2= (int *)malloc(N * sizeof(int)), w;
////    for(w=0;w<N;w++){ ptr1[w] = -1; ptr2[w] = -1; }
////                
////                
////    cl_mem  outLin = clCreateBuffer( clinfo.c,  CL_MEM_WRITE_ONLY, 2 * N * maxcols * sizeof(double), NULL, &status);
////    CLERR  
////    cl_mem  outCol = clCreateBuffer( clinfo.c,  CL_MEM_WRITE_ONLY, 2 * N * maxcols * sizeof(double), NULL, &status);
////    CLERR
////    cl_mem  idxOutLin = clCreateBuffer( clinfo.c,  CL_MEM_USE_HOST_PTR, N * maxcols * sizeof(int), ptr1, &status);
////    CLERR
////    cl_mem  idxOutCol = clCreateBuffer( clinfo.c,  CL_MEM_USE_HOST_PTR, N * maxcols * sizeof(int), ptr2, &status);
////    CLERR 
////    
////    cl_mem  mCol = clCreateBuffer( clinfo.c,  CL_MEM_USE_HOST_PTR, 2 * N * maxcols * sizeof(double), aCol, &status);
////    CLERR
////    cl_mem  idxCol = clCreateBuffer( clinfo.c,  CL_MEM_USE_HOST_PTR, N * maxcols * sizeof(int), aIdxCol, &status);
////    CLERR  
////
////    smatreqdev( a );
////    matSquare( &outLin, &idxOutLin, 
////               &outCol, &idxOutCol, 
////               a->extra, a->idxColMem, 
////               mCol, idxCol, 
////               maxcols, N );
////    smatrix_t * ret = (smatrix_t *) malloc( sizeof(smatrix_t) );
////    ret->nrow = a->nrow;
////    ret->ncol = a->ncol;
////    ret->maxcols = a->maxcols;
////    ret->type = T_COMPLEX;
////    ret->location = LOCDEV;
////    ret->extra = outLin;
////    ret->idxColMem = idxOutLin;
////    ret->idx_col = (int *) malloc(N * maxcols * sizeof(int));
////    ret->m = (double *) malloc(2 * N * maxcols * sizeof(double));
////    //smatreqhost( ret );    
////    int ii;       
////    /*for(ii=0;ii< N * maxcols;ii++)
////        printf("idx[%i]=%d [%f %f]\n", ii, ret->idx_col[ii],  ret->m[2*ii],ret->m[2*ii+1] );*/
////    object_t * out = (object_t *) malloc( sizeof( object_t ) );
////    type( *out ) = T_SMATRIX;
////    vvalue( *out ) = (void *) ret;
////    static void * rr[1];
////    rr[0] = (void *) out;
////    return rr;
//     return (void *) NULL;
//}
//
// void ** mat_transp( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    matrix_t * a = (matrix_t *) vvalue( *in[0] );
//    matreqdev( a );
//    object_t * out = (object_t *) malloc( sizeof( object_t ) );
//    matrix_t * r = (matrix_t *) malloc( sizeof(matrix_t) );
//    r->ncol = a->nrow;
//    r->nrow = a->ncol;
//    r->type = T_FLOAT;
//    r->extra = matTranspose( a->extra, a->ncol , a->nrow );
//    r->location = LOCDEV;
//    r->value.f = NULL;
//    type( *out ) = T_MATRIX;
//    vvalue( *out ) = (void *) r;
//    static void * ret[1];
//    ret[0] = (void *) out;
//    return ret;
//}
//
// void ** mat_transp_cpu( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    matrix_t * a = (matrix_t *) vvalue( *in[0] );
//    matreqhost( a );
//    object_t * out = (object_t *) malloc( sizeof( object_t ) );
//    matrix_t * r = (matrix_t *) malloc( sizeof(matrix_t) );
//    r->value.f = (double *) malloc(a->nrow * a->ncol * sizeof(double));
//    r->nrow = a->ncol;
//    r->ncol = a->nrow;
//    r->type = T_FLOAT;
//    int ii = 0, jj = 0;
//    for( ii = 0; ii < r->nrow; ii++ )
//        for( jj = 0; jj < r->ncol; jj++ )
//            r->value.f[ii*r->nrow+jj] = a->value.f[jj*a->nrow+ii];
//        
//    r->location = LOCHOS;
//    type( *out ) = T_MATRIX;
//    vvalue( *out ) = (void *) r;
//    static void * ret[1];
//    ret[0] = (void *) out;
//    return ret;
//}
//
//
// void ** mat_sub_cpu( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    matrix_t * a = (matrix_t *) vvalue( *in[0] );
//    matrix_t * b = (matrix_t *) vvalue( *in[1] );
//    matreqhost( a );matreqhost( b );
//    object_t * out = (object_t *) malloc( sizeof( object_t ) );
//    int len = b->ncol * b->nrow;    
//    matrix_t * r = (matrix_t *) malloc( sizeof(matrix_t) );
//    r->value.f = (double *) malloc(len * sizeof(double));
//    r->ncol = b->ncol;
//    r->nrow = b->nrow;
//    r->type = T_FLOAT;    
//    int k;
//    for(k=0;k<len;k++)
//        r->value.f[k] = b->value.f[k]-a->value.f[k];            
//    r->type = T_FLOAT;
//    r->location = LOCHOS;
//    type( *out ) = T_MATRIX;
//    vvalue( *out ) = (void *) r;
//    static void * ret[1];
//    ret[0] = (void *) out;
//    return ret;
//}
//
// void ** mat_add_cpu( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    matrix_t * a = (matrix_t *) vvalue( *in[0] );
//    matrix_t * b = (matrix_t *) vvalue( *in[1] );
//    matreqhost( a );matreqhost( b );
//    object_t * out = (object_t *) malloc( sizeof( object_t ) );
//    int len = b->ncol * b->nrow;    
//    matrix_t * r = (matrix_t *) malloc( sizeof(matrix_t) );
//    r->value.f = (double *) malloc(len * sizeof(double));
//    r->ncol = b->ncol;
//    r->nrow = b->nrow;
//    r->type = T_FLOAT;    
//    int k;
//    for(k=0;k<len;k++)
//        r->value.f[k] = a->value.f[k] + b->value.f[k];            
//    r->type = T_FLOAT;
//    r->location = LOCHOS;
//    type( *out ) = T_MATRIX;
//    vvalue( *out ) = (void *) r;
//    static void * ret[1];
//    ret[0] = (void *) out;
//    return ret;
//}
//
 void ** vec_mulsc( bridge_manager_t *m, int index, void ** i, int * status ) {
        object_t ** in = (object_t **) i;
        double scalar = 1.0;  
        if( type( *in[0] ) == T_FLOAT ) {
            scalar = fvalue( *in[0] );
        } else if( type( *in[0] ) == T_INT )
            scalar = (double) ivalue( *in[0] );
        else {
            // RUNTIME ERROR
        }
        vector_t * v = (vector_t *) vvalue( *in[1] );
        m->bridges[index].vecreqdev( v );
        
        vector_t * r = m->bridges[index].vector_new(v->len, T_FLOAT, 0, NULL);
        r->location = LOCDEV;
        
        r->extra = (void*)m->bridges[index].mulScalarVector_f( v->extra, scalar, v->len ); 
        if (status != NULL) {
            *status = 0;
        }
        return (void *) r;
}
 
 vector_t * vec_mul_complex_scalar ( bridge_manager_t *m, int index, complex_t * s, vector_t * a) {
        
        m->bridges[index].vecreqdev( a );
        
        vector_t * r = m->bridges[index].vector_new(a->len, T_COMPLEX, 0, NULL); //(vector_t *) malloc( sizeof( vector_t ) );
        r->location = LOCDEV;
        //apenas cpu
        // free( r->value.f);
        r->extra = (void*)m->bridges[index].mulComplexScalarVector_f( a->extra, s->re, s->im, a->len ); 
        
        return (void *) r;
}
 
 vector_t * mul_complex_scalar_complex_vec( bridge_manager_t *m, int index, complex_t * s, vector_t * a){
         
        m->bridges[index].vecreqdev( a );
        
        vector_t * r = m->bridges[index].vector_new(a->len, T_COMPLEX, 0, NULL); //(vector_t *) malloc( sizeof( vector_t ) );
        r->location = LOCDEV;
        //apenas cpu
        // free( r->value.f);
        r->extra = (void*)m->bridges[index].mulComplexScalarComplexVector_f( a->extra, s->re, s->im, a->len ); 
        
        return (void *) r;
}

 vector_t * mul_float_scalar_complex_vec( bridge_manager_t *m, int index, double d, vector_t * a){
         
        m->bridges[index].vecreqdev( a );
        
        vector_t * r = m->bridges[index].vector_new(a->len, T_COMPLEX, 0, NULL ); //(vector_t *) malloc( sizeof( vector_t ) );
        r->location = LOCDEV;
        //apenas cpu
        // free( r->value.f);
        
        r->extra = (void*)m->bridges[index].mulFloatScalarComplexVector_f( a->extra, d, a->len ); 
        
        return (void *) r;
}

// void ** vec_mulsc_cpu( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        int k;
//        double scalar = 1.0;  
//        if( type( *in[0] ) == T_FLOAT )
//            scalar = fvalue( *in[0] );
//        else if( type( *in[0] ) == T_INT )
//            scalar = (double) ivalue( *in[0] );
//        else {
//            // RUNTIME ERROR
//        }
//        vector_t * v = (vector_t *) vvalue( *in[1] );
//        vecreqhost( v );
//        object_t * out = (object_t *) malloc( sizeof( object_t ) );
//        vector_t * r = (vector_t *) malloc( sizeof( vector_t ) );
//        r->value.f = (double *) malloc(v->len * sizeof(double) );
//        for(k=0; k<v->len;k++) {
//            r->value.f[k] = scalar * v->value.f[k];
//        }
//        r->location = LOCHOS;
//        r->len = v->len;
//        r->type = T_FLOAT;
//        type( *out ) = T_VECTOR;
//        vvalue( *out ) = (void *) r;
//        static void * ret[1];
//        ret[0] = (void *) out;
//        return ret;
//}
//
// void ** mat_mulscrow   ( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    double scalar = 1.0;        
//    if( type( *in[1] ) == T_FLOAT )
//        scalar = fvalue( *in[1] );
//    else if( type( *in[1] ) == T_INT )
//        scalar = (double) ivalue( *in[1] );
//    else {
//         fprintf(stderr, "invalid scalar on 'mat_mulscrow'\n");
//        exit( 1 );
//    }
//    
//    int line = ivalue( *in[0] );
//    
//   
//    matrix_t * m = (matrix_t *) vvalue( *in[2] );
//    if( line > m->nrow ) {
//        fprintf(stderr, "invalid line on 'mat_mulscrow'\n");
//        exit( 1 );
//    }
//    
//    matreqdev( m );
//    line--;
//    mulScalarMatRow( m->extra, scalar, m->nrow, m->ncol, line );        
//    
//    void * itoclear[2];
//    itoclear[0] = i[0];
//    itoclear[1] = i[1];
//    clear_input(itoclear, 2);
//        
//    return NULL;    
//}
//
// void ** mat_mulsccol  ( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    double scalar = 1.0;        
//    if( type( *in[1] ) == T_FLOAT )
//        scalar = fvalue( *in[1] );
//    else if( type( *in[1] ) == T_INT )
//        scalar = (double) ivalue( *in[1] );
//    else {
//         // RUNTIME ERROR*
//    }
//    
//    int col = ivalue( *in[0] );
//    
//    
//    matrix_t * m = (matrix_t *) vvalue( *in[2] );
//    if( col > m->ncol ) {
//        fprintf(stderr, "invalid column on 'mat_mulsccol'\n");
//        exit( 1 );
//    }
//    matreqdev( m );
//    col--;
//    mulScalarMatCol( m->extra, scalar, m->nrow, m->ncol, col );        
//    
//    void * itoclear[2];
//    itoclear[0] = i[0];
//    itoclear[1] = i[1];
//    clear_input(itoclear, 2);
//        
//    return NULL;    
//}
 void ** mat_mulsc( bridge_manager_t *mg, int index, void ** i, int * status ) {
        object_t ** in = (object_t **) i;
        double scalar = 1.0;        
        if( type( *in[0] ) == T_FLOAT )
            scalar = fvalue( *in[0] );
        else if( type( *in[0] ) == T_INT )
            scalar = (double) ivalue( *in[0] );
        else {
            // RUNTIME ERROR
        }
        
        matrix_t * m = (matrix_t *) vvalue( *in[1] );
        mg->bridges[index].matreqdev( m );

        matrix_t * r = NULL;
        if( m->type == T_FLOAT ) {
            r = mg->bridges[index].matrix_new(m->nrow, m->ncol, T_FLOAT, 0, NULL);
            r->extra = mg->bridges[index].mulScalarVector_f( m->extra, scalar, m->nrow * m->ncol ); 
            r->location = LOCDEV;
        } else if( m->type == T_COMPLEX ) {
            r = mg->bridges[index].matrix_new(m->nrow, m->ncol, T_COMPLEX, 0, NULL);
            r->extra = mg->bridges[index].mulScalarVector_f( m->extra, scalar, 2 * m->nrow * m->ncol ); 
            r->location = LOCDEV;
        }
        
        clear_input(i, 2);
        if (status != NULL) {
            *status = 0;
        }
        return (void *) r;
}

matrix_t * mul_complex_scalar_complex_mat( bridge_manager_t *mg, int index, complex_t * s, matrix_t * m){
    matrix_t * r = NULL;
    mg->bridges[index].matreqdev( m );
    r = mg->bridges[index].matrix_new(m->nrow, m->ncol, T_COMPLEX, 0, NULL);
    r->extra = mg->bridges[index].mulComplexScalarComplexVector_f( m->extra, s->re, s->im, m->nrow * m->ncol ); 
    r->location = LOCDEV;

    return (void *) r;
 }

matrix_t * mul_complex_scalar_float_mat( bridge_manager_t *mg, int index, complex_t * s, matrix_t * m){
    matrix_t * r = NULL;
    mg->bridges[index].matreqdev( m );
    r = mg->bridges[index].matrix_new(m->nrow, m->ncol, T_COMPLEX, 0, NULL);
    r->extra = mg->bridges[index].mulComplexScalarVector_f( m->extra, s->re, s->im, m->nrow * m->ncol ); 
    r->location = LOCDEV;

    return (void *) r;
 }
 
// void ** mat_mulsc_cpu( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        int k;
//        double scalar = 1.0;        
//        if( type( *in[0] ) == T_FLOAT )
//            scalar = fvalue( *in[0] );
//        else if( type( *in[0] ) == T_INT )
//            scalar = (double) ivalue( *in[0] );
//        else {
//            // RUNTIME ERROR
//        }
//
//        matrix_t * m = (matrix_t *) vvalue( *in[1] );
//        matreqhost( m );
//        object_t * out = (object_t *) malloc( sizeof( object_t ) );
//        matrix_t * r = (matrix_t *) malloc( sizeof( matrix_t ) );
//        r->value.f = (double *) malloc(m->nrow * m->ncol * sizeof(double) );
//        // int k = 0;
//        for(k = 0; k < (m->nrow * m->ncol); k++ ) {
//            r->value.f[k] = scalar * m->value.f[k];
//        }
//        r->location = LOCHOS;
//        r->nrow = m->nrow;
//        r->ncol = m->ncol;
//        r->type = T_FLOAT;
//        type( *out ) = T_MATRIX;
//        vvalue( *out ) = (void *) r;
//        static void * ret[1];
//        ret[0] = (void *) out;
//        return ret;
//}
//
//
//

void print_smatrix(const smatrix_t* matrix) {
    printf("em %s: void print_smatrix(const smatrix_t* matrix)\n",__FILE__);
    //return;
    if (!matrix) {
        printf("Matrix is NULL.\n");
        return;
    }

    printf("Matrix (%p):\n", (void*)matrix);
    printf("  Rows: %d, Cols: %d, NNZ: %d\n", matrix->nrow, matrix->ncol, matrix->nnz);
    printf("  isPacked : %d\n", matrix->isPacked);
    printf("  type     : %d\n", matrix->type);
//    printf("  location : %u\n", matrix->location);
//    printf("  extra    : %p\n", matrix->extra);
//    printf("  idxColMem: %p\n", matrix->idxColMem);

    if (matrix->row_ptr) {
        printf("  row_ptr: ");
        for (int i = 0; i <= matrix->nrow; i++) {
            printf("%ld ", matrix->row_ptr[i]);
        }
        printf("\n");
    } else {
        printf("  row_ptr is NULL.\n");
    }

    if (matrix->col_idx) {
        printf("  col_idx: ");
        for (int i = 0; i < matrix->nnz; i++) {
            printf("%ld ", matrix->col_idx[i]);
        }
        printf("\n");
    } else {
        printf("  col_idx is NULL.\n");
    }

    int novoIndice;
    if (matrix->values) {
        printf("  values: ");
        for (int i = 0; i < matrix->nnz; i++) {
            novoIndice=i;
            if(matrix->type==T_COMPLEX) novoIndice=2*i;
            printf("%.4f ", matrix->values[novoIndice]);
            if(matrix->type==T_COMPLEX) printf("%.4f ", matrix->values[novoIndice+1]);
        }
        printf("\n");
    } else {
        printf("  values is NULL.\n");
    }
    return;

    if (matrix->smat) {
        printf("  smat:\n");
        for (int i = 0; i < matrix->nrow; i++) {
            printf("    Row %d: ", i);
            slist* current = matrix->smat[i];
            while (current) {
                printf("(col: %d, re: %.4f, im: %.4f) ", current->col, current->re, current->im);
                current = current->next;
            }
            printf("\n");
        }
    } else {
        printf("  smat is NULL.\n");
    }
}

void printIdxColMem(void* idxColMem, int size) {
    printf("idxColMem: ");
    int* arr = (int*)idxColMem;  // Cast to the expected type (e.g., int*)
    
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// void print_vectorT_hiperblas_std(vector_t * v_){
//	printf("BD, em %s: print_vectorT_hiperblas_std, ", __FILE__);
void print_vectorT(vector_t *v_) {
    if (v_ == NULL) { printf("BD, em %s: print_vectorT, vetor NULL\n", __FILE__); return; }

    int n = v_->len;
    if (n <= 0) { printf("BD, em %s: print_vectorT, vetor vazio\n", __FILE__); return; }

    //if (v_->extra == NULL) { printf("BD, em %s: print_vectorT, v_->extra é NULL\n", __FILE__); return; }

    printf("BD, em %s: print_vectorT, ", __FILE__); setvbuf(stdout, NULL, _IONBF, 0);

    //printf("\n  extra   (%p),  value.f (%p)\n",  v_->extra, v_->value.f);

    char formatoF[] = " %.2f";
    double *data = (double *) v_->value.f;
    if(data != NULL ) {
      printf("\nfrom v_->value.f [%d:%d]:", 0, n - 1);
    } else {
      data = (double *) v_->extra;
      printf("\nfrom v_->extra   [%d:%d]:", 0, n - 1);
    }
    // Detecta se é complexo — pode usar flag interna ou inferir
    int is_complex = (v_->type == T_COMPLEX); // (v_->is_complex != 0); // suponha que vector_t tenha um campo is_complex
    //printf("\nv_->type == T_COMPLEX, %d == %d\n", v_->type , T_COMPLEX); exit(2222);

    double sum = 0.0; int i;

    if (!is_complex) {
        // ---------- Vetor Real ----------
        if (n <= 20) {
            for (i = 0; i < n; i++) {
                sum += data[i] * data[i];
                printf(formatoF, data[i]);
            }
        } else {
            int tamFaixa = 5;
            for (i = 0; i < n; i++) {
                sum += data[i] * data[i];
                if (i < tamFaixa) printf(formatoF, data[i]);
                else if (i == tamFaixa) printf(" ...");
                else if (i >= n - tamFaixa) printf(formatoF, data[i]);
            }
        }
    } else {
        // ---------- Vetor Complexo ----------
        int n_complex = n / 1; // cada número tem parte real e imaginária
        printf("from v_->extra [0:%d]:", n_complex - 1);
        if (n_complex <= 10) {
            for (i = 0; i < n_complex; i++) {
                double re = data[2 * i], im = data[2 * i + 1];
                sum += re * re + im * im;
                //printf(" (%.3f %+ .3fi)", re, im);
        printf(" (%.3f %+.3fi)", re, im);
            }
        } else {
            int tamFaixa = 3;
            for (i = 0; i < n_complex; i++) {
                double re = data[2 * i], im = data[2 * i + 1];
                sum += re * re + im * im;
        if (i < tamFaixa) printf(" (%.3f %+.3fi)", re, im);
                else if (i == tamFaixa) printf(" ...");
                else if (i >= n_complex - tamFaixa) printf(" (%.3f %+.3fi)", re, im);
            }
        }
    }
    printf(", L2Norm = %.6f\n", sqrt(sum));
    return;
}


void allocate_result(smatrix_t *p, smatrix_t *d, smatrix_t *r){
    r->row_ptr = (long int*)    malloc((p->nrow + 1) * sizeof(long int));
    r->col_idx = (long int*)    malloc((d->nnz)      * sizeof(long int));
    //r->row_ptr = (int*)    malloc((p->nrow + 1) * sizeof(int));
    //r->col_idx = (int*)    malloc((d->nnz)      * sizeof(int));
    r->type    = d->type;

    if(r->type == T_COMPLEX)
        r->values  = (double*) malloc((2*d->nnz)   * sizeof(double));
    else
        r->values  = (double*) malloc((d->nnz)      * sizeof(double));

    r->nnz = d->nnz; // r has the same stricture as d

    if (!r->row_ptr || !r->col_idx || !r->values) {
        printf("Erro: Failing to allocate memory for r.\n");
        exit(0);
    }
}

/*
void computeRowptrU(const smatrix_t* S, const smatrix_t* C, smatrix_t* U) {
    U->row_ptr[0] = 0; // First row starts at index 0
    for (int i = 0; i < S->nrow; i++) {
        int permuted_row = S->col_idx[i];  // Get the row index in C
        int nnz_in_C_row = C->row_ptr[permuted_row + 1] - C->row_ptr[permuted_row];
        U->row_ptr[i + 1] = U->row_ptr[i] + nnz_in_C_row;
    }
}
void computeU(const smatrix_t* S, const smatrix_t* C, smatrix_t* U) {
   printf(" em computeU + Hiago, S->type = %d, C->type = %d, U->type = %d\n", S->type, C->type, U->type); //  exit(128+13+7);
    if (C->type == T_COMPLEX ) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < S->nrow; ++row) {
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
}
 
void permuteSparseMatrix(smatrix_t * S_,  smatrix_t * C_, smatrix_t * U_){
   printf("BD, em hiperblas-core/src/hiperblas_std.c: void ** permuteSparseMatrix( ... \n");
        //smatrix_t *U = mg->bridges[index].smatrix_new(C->nrow, C->nrow, C->type);
        //allocate_result(S, C, U);

//        printf("BD, em permuteSparseMatrix, CALL compute_U_row_ptr\n");
//        printf("BD, em permuteSparseMatrix, CALL print_smatrix(S_) \n"); print_smatrix(S_); 
//        printf("BD, em permuteSparseMatrix, CALL print_smatrix(C_) \n"); print_smatrix(C_); 
//        printf("BD, em permuteSparseMatrix, CALL print_smatrix(U_) \n"); print_smatrix(U_); 

        computeRowptrU(S_, C_, U_);
        //printf("BD, em permuteSparseMatrix, EXIT ANTES DO FIM!!!! _\n"); exit(128+13); // (void *) U_;
        printf("BD, em permuteSparseMatrix, CALL computeU\n"); computeU(S_, C_, U_);
        //printf("BD, em permuteSparseMatrix, CALL print_smatrix(U_) \n"); print_smatrix(U_); 
        //printf("BD, em permuteSparseMatrix, RETURN ANTES DO FIM!!!! _\n"); return; // (void *) U_;
        printf("BD, em permuteSparseMatrix, CALLed computeU, return'\n");
        //exit(128+13+11);
        return; // (void *) U_;
 
}
*/

 void  permuteSparseMatrix( bridge_manager_t *mg, int index, smatrix_t * S_, smatrix_t * C_, smatrix_t * U_ ) {
         mg->bridges[index].permuteSparseMatrix_f(S_, C_, U_);
            
  }

 void  matvec_mul3BD( bridge_manager_t *mg, int index, void ** i, int * status ) {
   printf("BD, em hiperblas-core/src/hiperblas_std.c: void ** matvec_mul3BD( bridge_manager_t *mg, int index, void ** i, int * status ) {\n");
    
        object_t ** in = (object_t **) i;
        vector_t * v = (vector_t *) vvalue( *in[1] );
        //printf("BD, em matvec_mul3BD: v->location= %d, LOCDEV = %d\n", v->location, LOCDEV);
        vector_t * r = (vector_t *) vvalue( *in[2] );
       
        //do I have to assume that it needs to be copied everytime?
	//BD if (v->location != LOCDEV){mg->bridges[index].vecreqdev( v );}
        
        if( type( *in[0] ) == T_MATRIX ) {        
            matrix_t * m = (matrix_t *) vvalue( *in[0] );
            r = mg->bridges[index].vector_new(m->nrow, m->type, 0, NULL );
            mg->bridges[index].vecreqdev( r );
            if (m->location != LOCDEV) {mg->bridges[index].matreqdev( m );}

            if(        m->type == T_FLOAT && v->type == T_FLOAT ) {
                mg->bridges[index].matVecMul3_f( m->extra, v->extra, r->extra, m->ncol, m->nrow );
//              r->location = LOCDEV; r->value.f = NULL; r->len = m->nrow; r->type = T_FLOAT;
            } else if( m->type == T_COMPLEX && v->type == T_COMPLEX ) {
                //BD mg->bridges[index].matVecMul3Complex_f( m->extra, v->extra, r->extra, m->ncol, m->nrow );
//              r->location = LOCDEV; r->value.f = NULL; r->len = m->nrow; r->type = T_COMPLEX;
            }
            if (status != NULL) { *status = 0; }
            return; // (void *) r;

        } else  if( type( *in[0] ) == T_SMATRIX ) {
            //printf(">\n");
            smatrix_t * m = (smatrix_t *) vvalue( *in[0] );
           // r = mg->bridges[index].vector_new(m->nrow, m->type, 0, NULL );
            mg->bridges[index].vecreqdev( r );
//            printf("BD, em matvec_mul3BD, Before call smatreqdev( m ), m->location: %d\n", m->location);
            if (m->location != LOCDEV) { mg->bridges[index].smatreqdev( m ); }
                    // m->values  volta em m->extra 
                    // m->col_idx volta em m->idxColMem, 
                    // m->crow_ptr não altera 
 //           printf("BD, em matvec_mul3BD, After  call smatreqdev( m ), m->location: %d\n", m->location);

            if(m->type == T_FLOAT && v->type == T_FLOAT ) {
                // idxColMem é nome de variavel de do neblinaa, implementacao em abandono
                mg->bridges[index].sparseVecMul_f( v->extra, r->extra, m->extra, m->row_ptr, m->idxColMem, m->nrow, m->nnz );
//              r->location = LOCDEV; r->value.f = NULL; r->len = m->nrow; r->type = T_FLOAT;
            } else if( m->type == T_COMPLEX && v->type == T_COMPLEX ) {
                //printIdxColMem(m->idxColMem, m->nnz);
                mg->bridges[index].sparseComplexVecMul_f( v->extra, r->extra, m->extra, m->row_ptr, m->idxColMem, m->nrow, m->nnz );
//              r->location = LOCDEV; r->value.f = NULL; r->len = m->nrow; r->type = T_COMPLEX;
            } else {
		    // bloco vazio
            }

            if (status != NULL) { *status = 0; }
            //printf("BD,   em matvec_mul3BD, RETURN (void *) r  \n");
            return;
            
        } else if(  type( *in[0] ) == T_RMATRIX ) {
//                rmatrix_t * m = (rmatrix_t *) vvalue( *in[1] );
//                r->extra = (void*)rmatVecMul3Complex( m, v->extra, m->ncol, m->nrow );
//                r->location = LOCDEV;
//                r->value.f = NULL;
//                r->len = m->nrow;
//                r->type = T_COMPLEX;
//                type( out ) = T_VECTOR;
//                vvalue( out ) = (void *) r;
//                static void * ret[1];
//                clear_input(i, 2);
//                ret[0] = (void *) &out;
//                return ret;
                  if (status != NULL) { *status = -1; }
                  return;// (void **)NULL;   
        } else {
            if (status != NULL) { *status = -1; }
            return; // (void **)NULL;   
        }             
}

// void ** matvec_mul_cpu( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        vector_t * v = (vector_t *) vvalue( *in[0] );
//        matrix_t * m = (matrix_t *) vvalue( *in[1] );
//        vecreqhost( v ); matreqhost( m );
//        object_t out;
//        vector_t * r = matvec_multiply( m, v );
//        r->location = LOCHOS;
//        type( out ) = T_VECTOR;
//        vvalue( out ) = (void *) r;
//        static void * ret[1];
//        clear_input( i, 2 );
//        ret[0] = (void *) &out;
//        return ret;
//}
//
// void ** smatvec_mul_cpu( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        vector_t * v = (vector_t *) vvalue( *in[0] );
//        smatrix_t * m = (smatrix_t *) vvalue( *in[1] );
//        vecreqhost( v ); smatreqhost( m );
//        
//        object_t out;
//        vector_t * r = smatvec_multiply( m, v );
//        r->location = LOCHOS;
//        type( out ) = T_VECTOR;
//        vvalue( out ) = (void *) r;
//        static void * ret[1];
//        clear_input( i, 2 );
//        ret[0] = (void *) &out;
//        return ret;
//}
//
//
//
//void ** init ( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    if( type( *in[1] ) == T_VECTOR && (type(*in[0]) == T_FLOAT || type(*in[0]) == T_INT) ) {
//        vector_t * v = (vector_t *) vvalue( *in[1] );
//        vecreqhost( v );
//        object_t * s = in[0];
//        int i = 0;
//        double f = (type(*in[0]) == T_FLOAT) ?  s->value.f : s->value.i;
//        #pragma omp parallel for
//        for( i=0;i < v->len; i++ )
//            v->value.f[i] = f; 
//    } else  if( type( *in[1] ) == T_MATRIX && (type(*in[0]) == T_FLOAT || type(*in[0]) == T_INT) ) {
//        matrix_t * v = (matrix_t *) vvalue( *in[1] );
//        matreqhost( v );
//        object_t * s = in[0];
//        int i = 0;
//        double f = (type(*in[0]) == T_FLOAT) ?  s->value.f : s->value.i;
//        int size = v->nrow * v->ncol;
//        #pragma omp parallel for
//        for( i=0;i <size; i++ )
//            v->value.f[i] = f;
//    } else {
//        fprintf(stderr, "invalid argument on init\n");
//        exit( 1 );
//    }
//    return (void **) NULL;
//}
//
// void ** toint ( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t out;
//    type( out ) = T_INT;
//    if( type( *in[0] ) == T_INT )
//        ivalue( out ) = ivalue( *in[0] );
//    else if( type( *in[0] ) == T_FLOAT )
//        ivalue( out ) = fvalue( *in[0] );
//    else if( type( *in[0] ) == T_STRING ) {
//        ivalue( out ) = atoi( svalue( *in[0] ));
//    } else {
//        fprintf(stderr, "invalid use of 'toint' function\n");
//        exit( 1 );
//    }
//    
//    static void * ret[1];
//    clear_input( i, 1 );
//    ret[0] = (void *) &out;
//    return ret;
//}
//
// void ** todouble ( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t * out = (object_t *) malloc( sizeof( object_t ) );
//    type( *out ) = T_FLOAT;
//    if( type( *in[0] ) == T_INT )
//        fvalue( *out ) = ivalue( *in[0] );
//    else if( type( *in[0] ) == T_FLOAT )
//        fvalue( *out ) = fvalue( *in[0] );
//    else if( type( *in[0] ) == T_STRING )
//        fvalue( *out ) = atof( svalue( *in[0] ));
//    else {
//        fprintf(stderr, "invalid use of 'todouble' function\n");
//        exit( 1 );
//    }
//    
//    static void * ret[1];
//    ret[0] = (void *) out;
//    return ret;
//}
//
// void ** tostr ( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t out;
//    type( out ) = T_STRING;
//    if( type( *in[0] ) == T_INT ) {
//        svalue( out ) = new_str( 32 );
//        sprintf( svalue( out ), "%d", ivalue( *in[0] ) );  
//    } else if( type( *in[0] ) == T_FLOAT ) {
//        svalue( out ) = new_str( 64 );
//        sprintf( svalue( out ), "%f", fvalue( *in[0] ) );
//    } else {
//        // runerror("invalid use of 'int' function");
//        fprintf(stderr, "invalid use of 'tostr' function\n");
//        exit( 1 );
//    }
//    
//    static void * ret[1];
//    clear_input(i,1);
//    ret[0] = (void *) &out;
//    return ret;
//}
//
// void ** tostr2 ( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t out;
//    type( out ) = T_STRING;
//    if( type( *in[1] ) == T_INT ) {
//        svalue( out ) = new_str( 32 );
//        char buf[6];
//        sprintf( buf, "%%0%dd", ivalue( *in[0] ) );
//        sprintf( svalue( out ), buf, ivalue( *in[1] ) );  
//    } else if( type( *in[1] ) == T_FLOAT ) {
//        svalue( out ) = new_str( 64 );
//        sprintf( svalue( out ), "%f", fvalue( *in[1] ) );
//    } else {
//        // runerror("invalid use of 'int' function");
//        fprintf(stderr, "invalid use of 'tostr' function\n");
//        exit( 1 );
//    }
//    
//    static void * ret[1];
//    clear_input(i,2);
//    ret[0] = (void *) &out;
//    return ret;
//}
//
//// void ** complex_new ( void ** i, int * status ) {
////    object_t ** in = (object_t **) i;
////    object_t  out;// = (object_t *) malloc( sizeof( object_t ) );
////    
////    double re = 0;
////    double im = 0;
////    
////    if( type( *in[1] ) == T_INT )
////        re = ivalue( *in[1] );
////    else if( type( *in[1] ) == T_FLOAT )
////        re = fvalue( *in[1] );
////    else {
////        // runerror("invalid use of 'int' function");
////        fprintf(stderr, "invalid use of 'complex' function\n");
////        exit( 1 );
////    }
////    
////    if( type( *in[0] ) == T_INT )
////        im = ivalue( *in[0] );
////    else if( type( *in[0] ) == T_FLOAT )
////        im = fvalue( *in[0] );
////    else {
////        // runerror("invalid use of 'int' function");
////        fprintf(stderr, "invalid use of 'complex' function\n");
////        exit( 1 );
////    }
////    
////    complex_t * res = (complex_t *) malloc( sizeof(complex_t) );
////    res->im = im;
////    res->re = re;
////    
////    type( out ) = T_COMPLEX;
////    vvalue( out ) = (void *) res;
////    
////    //printf("Complex NEW\n");
////    clear_input( i, 2);
////    static void * ret[1];
////    ret[0] = (void *) &out;
////    return ret;
////}
//
// void ** complex_real( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t  out;
//    
//    if( type( *in[0] ) != T_COMPLEX ) {
//        
//       fprintf(stderr, "invalid use of 'real' function\n");
//       exit( 1 );
//    }
//    
//    complex_t * r = (complex_t *) vvalue( *in[0] ); 
//    
//    type( out ) = T_FLOAT;
//    fvalue( out ) = r->re;
//    
//    clear_input(i,1);
//        
//    static void * ret[1];
//    ret[0] = (void *) &out;
//    return ret;
//}
//
// void ** complex_imag( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t  out;// = (object_t *) malloc( sizeof( object_t ) );
//    
//    if( type( *in[0] ) != T_COMPLEX ) {
//        
//       fprintf(stderr, "invalid use of 'imag' function\n");
//       exit( 1 );
//    }
//    
//    complex_t * r = (complex_t *) vvalue( *in[0] ); 
//    
//    type( out ) = T_FLOAT;
//    fvalue( out ) = r->im;
//    
//    clear_input(i,1);
//    static void * ret[1];
//    ret[0] = (void *) &out;
//    return ret;
//}
//
// void ** complex_conj( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t  out;// = (object_t *) malloc( sizeof( object_t ) );
//    
//    
//    complex_t * r = (complex_t *) vvalue( *in[0] ); 
//    
//    complex_t * res = (complex_t *) malloc( sizeof(complex_t) );
//    res->im = -r->im;
//    res->re = r->re;
//    
//    type( out ) = T_COMPLEX;
//    vvalue( out ) = (void *) res;
//    
//    
//    clear_input(i,1);
//    static void * ret[1];
//    ret[0] = (void *) &out;
//    return ret;
//}
//
// void ** hiperblas_at ( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t out;
//    type( out ) = T_STRING;
//    int idx = ivalue( *in[0] );
//    int len = strlen( svalue( *in[1] ) );
//    
//    if( (idx - 1) < 0 || (idx - 1) >= len ) {
//        svalue( out ) = new_str(1);
//        svalue( out )[0] = 0;
//    } else {
//        svalue( out ) = new_str(2);
//        svalue( out )[0] = svalue( *in[1] )[idx-1];
//        svalue( out )[1] = 0;
//    }
//    
//    static void * ret[1];
//    clear_input(i,2);
//    ret[0] = (void *) &out;
//    return ret;
//}
//
// void ** hiperblas_upper( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t out;
//    int k = 0;
//    type( out ) = T_STRING;
//    int len = strlen( svalue( *in[0] ) );
//    svalue( out ) = new_str( len + 1 );
//    
//    for( k = 0; k < len; k++ )
//        svalue( out )[k] = toupper(  svalue( *in[0] )[k] );
//    svalue( out )[len] = 0;    
//    
//    static void * ret[1];
//    clear_input(i,1);
//    ret[0] = (void *) &out;
//    return ret;
//}
//
// void ** hiperblas_lower( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t out;
//    int k = 0;
//    type( out ) = T_STRING;
//    int len = strlen( svalue( *in[0] ) );
//    svalue( out ) = new_str( len + 1 );
//    
//    for( k = 0; k < len; k++ )
//        svalue( out )[k] = tolower(  svalue( *in[0] )[k] );
//    svalue( out )[len] = 0;    
//    
//    static void * ret[1];
//    clear_input(i,1);
//    ret[0] = (void *) &out;
//    return ret;
//}
//
// void ** hiperblas_type( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t out;
//    int k = 0;
//    type( out ) = T_STRING;
//    char tmp[256];
//    hiperblas_strtype( type( *in[0] ), tmp );
//    int len = strlen( tmp );
//    
//    svalue( out ) = new_str( len + 1 );
//    strcpy(svalue( out ), tmp );
//    svalue( out )[len] = 0;
//    
//    static void * ret[1];
//    clear_input(i,1);
//    ret[0] = (void *) &out;
//    return ret;
//}
///*
// void ** vec_add_off    ( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t out;
//    int offset = ivalue( *in[0] );
//    
//    vector_t * a = (vector_t *) vvalue( *in[1] );    
//    vecreqhost( a );
//    vector_t * r = (vector_t *) malloc( sizeof( vector_t ) );
//    int parts = a->len / offset;
//    r->value.f = (double *) malloc( offset * sizeof(double) );
//    r->len = offset;
//    r->type = T_FLOAT;
//    r->location = LOCHOS;
//    int j, l;    
//    type( out ) = T_VECTOR;
//    vvalue( out ) = r;
//    double s = 0;
//    #pragma omp parallel for
//    for(j=0; j < r->len; j++ ) {
//        s = 0;
//        for(l=0; l < parts; l++ ) { 
//            s += a->value.f[j+l*offset];   
//        }
//        r->value.f[j] = s;
//    }
//    static void * ret[1];
//    clear_input(i,2);
//    ret[0] = (void *) &out;
//    return ret;
//}*/
//
 object_t ** convertToObject2(int n, vector_t * a) {
    object_t ** in;
    in = (object_t **) malloc(2 * sizeof(object_t *));

    in[0] = (object_t *) malloc(sizeof(object_t));
    ivalue( *in[0] ) = n; in[0]->type = T_INT;

    in[1] = (object_t *) malloc(sizeof(object_t));
    vvalue( *in[1] ) = a; in[1]->type = T_VECTOR;
    
    return in;
 }

 void ** vec_add_off    ( bridge_manager_t *m, int index, void ** i, int * status ) {
    
        object_t ** in = (object_t **) i;
        int offset = ivalue( *in[0] );
        vector_t * a = (vector_t *) vvalue( *in[1] );
        int parts = a->len / offset;
        
        vector_t * r = m->bridges[index].vector_new(offset, T_FLOAT, 0, NULL);
        //apenas cpu
        // free( r->value.f);
        m->bridges[index].vecreqdev( a ); 
        m->bridges[index].vecreqdev( r );

        r->extra = (void*)m->bridges[index].vecAddOff_f( a->extra, offset, parts ); 

        clear_input( i, 2 );
        if (status != NULL) {
            *status = 0;
        }
        return (void *) r;
}
//
// void ** vec_add_off2   ( void ** i, int * status ) {
//    
//        object_t ** in = (object_t **) i;
//        vector_t * a = (vector_t *) vvalue( *in[0] );
//        vecreqdev( a );
//        object_t out; 
//        vector_t * r = (vector_t *) malloc( sizeof( vector_t ) );
//       
//        r->extra = (void*)vecAddOff2( a->extra, a->len ); 
//        
//        r->len = a->len/2;
//        r->type = T_FLOAT;
//        r->location = LOCDEV;
//        r->value.f = NULL;
//        type( out ) = T_VECTOR;
//        vvalue( out ) = (void *) r;
//        clear_input(i,1);    
//        static void * ret[1];
//        ret[0] = (void *) &out;
//        return ret;
//}
//
// void ** hiperblas_ludecomp( void ** i, int * status ) {
//    object_t ** in = (object_t **) i;
//    object_t out;
//    
//    matrix_t * m = (matrix_t *) vvalue( *in[0] );
//    matreqdev( m );
//    luDecomp( m->extra, m->nrow );  
//    vvalue( out ) = (void *) m;
//    m->location = LOCDEV;
//    m->type = T_FLOAT;
//    out.type = T_MATRIX; 
//    // clear_input(i,1);
//    static void * ret[1];
//    ret[0] = (void *) &out;
//    return ret;
//}
// void ** hiperblas_solve( void ** I, int * status ) {
//    object_t ** in = (object_t **) I;
//    object_t out;
//    
//    matrix_t * A = (matrix_t *) vvalue( *in[1] );
//    vector_t * b = (vector_t *) vvalue( *in[0] );
//    matrix_t * M = (matrix_t *) malloc( sizeof(matrix_t) );
//    M->value.f = (double *) malloc(A->nrow * A->ncol * sizeof(double));
//    int n = A->nrow, i, j;
//    matreqhost( A );
//    #pragma omp parallel for
//    for( i = 0; i < A->nrow * A->ncol ; i++){ M->value.f[i] = A->value.f[i]; }
//    
//    // memcpy(M->value.f, A->value.f, sizeof(double) * A->nrow * A->ncol );
//    M->nrow = A->nrow;
//    M->ncol = A->ncol;
//    M->location = LOCHOS; 
//    matreqdev( M ); vecreqhost( b );
//    luDecomp( M->extra, M->nrow );  
//    matreqhost( M );
//    
//    double * x = (double *) malloc(A->nrow  * sizeof(double));
//    double * y = (double *) malloc(A->nrow  * sizeof(double));
//    // Forward solve Ly = b
//    for (i = 0; i < n; i++) {
//        y[i] = b->value.f[i];
//        for (j = 0; j < i; j++) {
//          y[i] -= M->value.f[i*M->nrow + j] * y[j];
//        }
//        //y[i] /= M->value.f[i*M->nrow + i];
//    }
//    // Backward solve Ux = y
//    for (i = n - 1; i >= 0; i--) {
//        x[i] = y[i];
//        for (j = i + 1; j < n; j++) {
//          x[i] -= M->value.f[i*M->nrow + j] * x[j];
//        }
//        x[i] /= M->value.f[i*M->nrow + i];
//    }
//    // FREE Y
//    free( y );
//    // FREE
//    //clReleaseMemObject( M->mem );  
//    if( M && M->value.f ) {
//        free( M->value.f );
//        free( M );
//    }
//    
//    
//    
//    vector_t * r = (vector_t *) malloc( sizeof( vector_t ) );
//    r->value.f = x;
//    r->len = A->nrow;   
//    vvalue( out ) = (void *) r;
//    r->location = LOCHOS;
//    r->type = T_FLOAT;
//    out.type = T_VECTOR; 
//    clear_input(I,2);
//    static void * ret[1];
//    ret[0] = (void *) &out;
//    return ret;
//}
//
//
//
// void ** hiperblas_list_new( void ** i, int * status ) {
//   object_t out;
//   type( out ) = T_LIST;
//   out.value.v = (void *) 0;
//   static void * ret[1];
//   ret[0] = (void *) &out;
//   clear_input( i, 0 );   
//   return ret;
//}
// void ** hiperblas_list_append( void ** i, int * status ) {
//        object_t ** in = (object_t **) i;
//        list_t * lst = (list_t *) vvalue( *in[1] );
//        object_t to_app = *in[0]; 
//        
//        lst = list_append( lst, to_app );
//        object_t out;
//   
//       type( out ) = T_LIST;
//       vvalue( out ) = lst;
//       static void * ret[1];
//       ret[0] = (void *) &out;
//       clear_input( i, 2 );
//       return ret;
//}
// void ** hiperblas_list_get( void ** i, int * status ) {
//       object_t ** in = (object_t **) i;
//       list_t * lst = (list_t *) vvalue( *in[1] );
//       object_t input = *in[0];
//       int idx = ivalue(  input );
//       
//       object_t out = list_get( lst, idx );
//       
//       static void * ret[1];
//       ret[0] = (void *) &out;
//       clear_input( i, 2 );
//       return ret;
// 
//}
////extern int nkernelsRmat;
//// void ** hiperblas_rmatrix( void ** i, int * status ) {
////       object_t ** in = (object_t **) i;
////       int ncol = ivalue( *in[0] );
////       int nrow = ivalue( *in[1] );
////       char * getij = svalue(  *in[2] ); 
////       
////       char nkernel[4096];
////       char kernelname[1024];
////       int id = 0, ii;
////       for(ii=0;ii<32;ii++)
////            if( rmatstat[ii] != 0 ){
////                id = ii;
////                break;
////            }
////       sprintf( kernelname, RMATKERNAME, id );
////       sprintf( nkernel, RMATKER, id, getij );
////    //   printf("%s\n", nkernel );       
////       cl_kernel kernel;
////       rmatrix_t * rm = (rmatrix_t *) malloc( sizeof( rmatrix_t) );
////       rm->ncol = ncol;
////       rm->nrow = nrow;
////       rm->id = id;
////       strcpy( strkernelsRmat[id], nkernel );
////       nkernelsRmat++;   
////       rmatstat[id] = 1; 
////       InitCLEngine(0); //fixed for now, this function is not exposed to Python yet
////       cl_int st;
////       rm->kernel = clCreateKernel (clinfo.p, kernelname, &st);
////       
////       object_t out;
////       type( out ) = T_RMATRIX;
////       vvalue( out ) = (void *) rm;
////       static void * ret[1];
////       clear_input(i, 3);
////       ret[0] = (void *) &out;
////       return ret;
////       
////}
////
////
////


