#include "libhiperblas.h"
#include <stdio.h>
#include <stdlib.h>

vector_t *vector_new(int len, data_type type, int initialize, void *data) {
// chatGPT em 18.11.2025, bidu
    vector_t *ret     = malloc(sizeof(vector_t));
    if (!ret) return NULL;
    ret->type         = type;
    ret->len          = len;
    ret->location     = LOCHOS;
    ret->extra        = NULL; // esta relacionado com GPU, ou device de modo geral
    ret->externalData = (data != NULL);
    // If user provided external data ---------------------------
    if (data != NULL) {
        switch (type) {
        case T_INT:
            ret->value.i = (int *)    data;
            break;
        case T_FLOAT:
        case T_COMPLEX:
            ret->value.f = (double *) data;
            break;
        }
        return ret;
    } // if (data != NULL) {

    // If we need to allocate internal data -----------------------
    if (initialize) {
        size_t bytes = 0;
        switch (type) {
        case T_INT:
            bytes = len * sizeof(int);
            ret->value.i = malloc(bytes);
            if (!ret->value.i) { free(ret); return NULL; }
            break;
        case T_FLOAT:
            bytes = len * sizeof(double);
            ret->value.f = malloc(bytes);
            if (!ret->value.f) { free(ret); return NULL; }
            break;
        case T_COMPLEX:
            // 2 doubles: real + imag
            bytes = 2 * len * sizeof(double);
            ret->value.f = malloc(bytes);
            if (!ret->value.f) { free(ret); return NULL; }
            break;
        }
    } else {
        // Not initializing → just set correct union field to NULL
        switch (type) {
        case T_INT:
            ret->value.i = NULL; break;
        case T_FLOAT:
        case T_COMPLEX:
            ret->value.f = NULL; break;
        }
    }
    return ret;
}


void vector_delete(vector_t *v)
{
// chatGPT em 18.11.2025, bidu
    printf("BD, ATENCAO, em %s: void vector_delete( vector_t * v ), NO FREE! {\n", __FILE__); // _NAME__);
    if (!v) return;

    /*
    printf("vector_delete: INICIO\n");
    printf("  type=%d, externalData=%d\n", v->type, v->externalData);
    printf("  value.f = %p\n", v->value.f);
    printf("  value.i = %p\n", v->value.i);
    printf("  extra   = %p\n", v->extra);
    */

    void *data_ptr = NULL;

    // Definimos o ponteiro de dados principal do vector
    switch (v->type) {
    case T_INT:      data_ptr = v->value.i; break;
    case T_FLOAT:
    case T_COMPLEX: data_ptr = v->value.f; break;
    default: break;
    }

    int extra_is_alias = (v->extra == data_ptr);

    // --- LIBERAR DADOS PRINCIPAIS ---
    if (!v->externalData && data_ptr != NULL) {
      //  printf("  freeing data_ptr (%p)\n", data_ptr);
        free(data_ptr);
        data_ptr = NULL;

        // Zerar corretamente o union
        if (v->type == T_INT) v->value.i = NULL;
        else v->value.f = NULL;
    }

    // --- LIBERAR EXTRA ---
    if (v->extra != NULL) {
        if (!extra_is_alias) {
     //       printf("  freeing extra (%p)\n", v->extra);
            free(v->extra);
        } else {
    //        printf("  extra == data_ptr → NÃO liberar novamente!\n");
        }
        v->extra = NULL;
    }

    //printf("  freeing struct v (%p)\n", v);
    free(v);
}

/* bidu
vector_t * vector_new00( int len, data_type type, int initialize, void * data ) {
    vector_t * ret = (vector_t *) malloc( sizeof( vector_t ) );
    if (initialize && data == NULL) {
        if( type == T_INT ) {
            ret->value.i = (int *)    malloc(     len * sizeof( int    ) );
        } else if( type == T_FLOAT )
            ret->value.f = (double *) malloc(     len * sizeof( double ) ); 
        else if( type == T_COMPLEX )
            ret->value.f = (double *) malloc( 2 * len * sizeof( double ) );
        ret->externalData = 0;
    } else if (data != NULL) {
        ret->value.f = (double *)data;
        ret->externalData = 1;
    } else {
        ret->value.f = NULL;
        ret->externalData = 0;
    }
    ret->type      = type;
    ret->len       = len;
    ret->location  = LOCHOS;
    ret->extra       = NULL;
    return ret;
}
void vector_delete00( vector_t * v ) {
    printf("BD, ATENCAO, em %s: void vector_delete( vector_t * v ), NO FREE! {\n", __FILE__); // _NAME__);
    return;
    if (v != NULL) {
        if (v->value.f != NULL && v->externalData == 0) {
            free(v->value.f);
        }
        if (v->extra != NULL) {  // No need to check externalData for extra
            free(v->extra);
        }
        free(v);
    }
}
*/

void vecreqhost( vector_t * v ) {
//    if (v->value.f != NULL) {
//    printf("vecreqhost 1\n");
//        free (v->value.f);
//    printf("vecreqhost 2\n");
//    }
    if (v->location == LOCHOS) return;
    v->location  = LOCHOS;
    v->value.f = v->extra;
    v->extra = NULL;
}

void vecreqdev ( vector_t * v ) {
//    if (v->extra != NULL) {
//    printf("vecreqdev 1\n");
//        free (v->extra);
//    printf("vecreqdev 2\n");
//    }
    if (v->location == LOCDEV) return;

    v->location  = LOCDEV;
    // printf("v->value.f %p\n",v->value.f);
    // printf("v->extra %p\n",v->extra);

    v->extra = v->value.f;
    v->value.f = NULL;
    // printf("v->value.f %p\n",v->value.f);
    // printf("v->extra %p\n",v->extra);
    // printf("vecreqdev %p\n",v->value.f);
}

