#include "libhiperblas.h"
#include <stdio.h>
#include <stdlib.h>

slist * slist_add( slist * l, int col, double re, double im ) {
    slist * nlist = (slist *) malloc( sizeof(slist) );
    nlist->col = col;
    nlist->re = re;
    nlist->im = im;
    nlist->next = l;
    return nlist;
}


void slist_clear( slist * l ) {
    if( l == NULL )
        return;    
    do {    
        slist * tmp = l->next;
        free( l );
        l = tmp;                
    } while( l != NULL );
}

void smatrix_pack(smatrix_t * m) {

    m->row_ptr = (long long int *) malloc((m->nrow + 1) * sizeof(long long int));
    m->col_idx = (long long int *) malloc(m->nnz * sizeof(long long int));
    //m->row_ptr = (int *) malloc((m->nrow + 1) * sizeof(int)); //ou long long int*
    //m->col_idx = (int *) malloc(m->nnz * sizeof(int)); //ou long long int*

    m->values = (double *) malloc(m->nnz * sizeof(double));
    
    int nnz_counter = 0;
    m->row_ptr[0] = 0;
    for (int i = 0; i < m->nrow; i++) {
        slist *tmp = m->smat[i];
        while (tmp != NULL) {
            m->col_idx[nnz_counter] = tmp->col;
            m->values[nnz_counter] = tmp->re;
            nnz_counter++;
            tmp = tmp->next;
        }
        m->row_ptr[i + 1] = nnz_counter;
        slist_clear(m->smat[i]);
        m->smat[i] = NULL;
    }
    m->isPacked = 1;
    //print_smatrix(m) ;
}

void smatrix_pack_complex(smatrix_t * m) {
    m->row_ptr = (long long int *) malloc((m->nrow + 1) * sizeof(long long int)); 
    m->col_idx = (long long int *) malloc(m->nnz * sizeof(long long int)); 
    //m->row_ptr = (int *) malloc((m->nrow + 1) * sizeof(int));
    //m->col_idx = (int *) malloc(m->nnz * sizeof(int));
    m->values = (double *) malloc(2 * m->nnz * sizeof(double));
    
    int nnz_counter = 0;
    m->row_ptr[0] = 0;
    for (int i = 0; i < m->nrow; i++) {
        slist *tmp = m->smat[i];
        while (tmp != NULL) {
            m->col_idx[nnz_counter] = tmp->col;
            m->values[2 * nnz_counter] = tmp->re;
            m->values[2 * nnz_counter + 1] = tmp->im;
            nnz_counter++;
            tmp = tmp->next;
        }
        m->row_ptr[i + 1] = nnz_counter;
        slist_clear(m->smat[i]);
        m->smat[i] = NULL;
    }
    m->isPacked = 1;
}

void smatrix_set_real_value(smatrix_t * m, int i, int j, double r) {
    //printf(" src/libhiperblas-cpu-bridge-smatrix.c: void smatrix_set_real_value(smatrix_t * m, int i, int j, double r)\n");
     //printf(" r = %f\n", r);

    if (i < 0 || i >= m->nrow || j < 0 || j >= m->ncol) {
        printf("Invalid index on loading sparse matrix: row %d, col %d\n", i, j);
        exit(-1);
    }
    
    slist *new_node = slist_add(m->smat[i], j, r, 0.0);
    m->smat[i] = new_node;
    m->nnz++;
}

void smatrix_set_complex_value(smatrix_t * m, int i, int j, double r, double im) {
    if (i < 0 || i >= m->nrow || j < 0 || j >= m->ncol) {
        printf("Invalid index on loading sparse matrix: row %d, col %d\n", i, j);
        exit(-1);
    }
    
    slist *new_node = slist_add(m->smat[i], j, r, im);
    m->smat[i] = new_node;
    m->nnz++;
}

smatrix_t * smatrix_new(int nrow, int ncol, data_type type) {
    printf("BD, em %s, %s\n", __FILE__, __func__); 

    //smatrix_t *smatrix = (smatrix_t *) malloc(sizeof(smatrix_t));

    smatrix_t *smatrix =  (smatrix_t *)  calloc(1, sizeof(smatrix_t));

    smatrix->ncol = ncol;
    smatrix->nrow = nrow;
    smatrix->type = type;
    smatrix->nnz = 0;
    smatrix->location = 0;
    
    smatrix->row_ptr = NULL;
    smatrix->col_idx = NULL;
    smatrix->values  = NULL;
    smatrix->extra   = NULL;
    smatrix->isPacked = 0;
    return smatrix;

    smatrix->smat = (slist **) calloc(nrow, sizeof(slist *));

}

void smatrix_t_clear(smatrix_t *smatrix) {
    if (smatrix != NULL) {
        free(smatrix->values);
        free(smatrix->col_idx);
        free(smatrix->row_ptr);
        for (int i = 0; i < smatrix->nrow; i++) {
            slist_clear(smatrix->smat[i]);
        }
        free(smatrix->smat);
        free(smatrix);
    }
}

void smatrix_load_double(smatrix_t *m, FILE *f) {
    double e;
    int i, j;
    while (fscanf(f, "%d %d %lf", &i, &j, &e) != EOF) {
        i--; j--;
        if (i < 0 || i >= m->nrow || j < 0 || j >= m->ncol) {
            printf("Invalid index on loading sparse matrix\n");
            exit(-1);
        }
        m->smat[i] = slist_add(m->smat[i], j, e, 0.0);
        m->nnz++;
    }
    fclose(f);
}

void smatrix_load_complex(smatrix_t *m, FILE *f) {
    double re, im;
    int i, j;
    while (fscanf(f, "%d %d %lf %lf", &i, &j, &re, &im) != EOF) {
        i--; j--;
        if (i < 0 || i >= m->nrow || j < 0 || j >= m->ncol) {
            printf("Invalid index on loading sparse matrix\n");
            exit(-1);
        }
        m->smat[i] = slist_add(m->smat[i], j, re, im);
        m->nnz++;
    }
    fclose(f);
}

void smatreqhost(smatrix_t *m) {
    printf("em hiperblas-core/src/libhiperblas-cpu-bridge-smatrix.c: void smatreqhost(smatrix_t *m) {\n");
    if (m->location != LOCHOS) {
        m->location  = LOCHOS;
        m->col_idx   = m->idxColMem;  // Transfer idxColMem to idx_col
        m->values    = m->extra;  // Transfer extra to values
        m->extra     = NULL;  // Clear extra
        m->idxColMem = NULL;  // Clear idxColMem
    }
}

void smatreqdev(smatrix_t *m) {
       printf("em hiperblas-core/src/libhiperblas-cpu-bridge-smatrix.c: void smatreqdev(smatrix_t *m) {\n");
//       printf("em smatreqdev, inicio\n");

       //print_smatrix(m); 
       //printf("em hiperblas-core/src/libhiperblas-cpu-bridge-smatrix.c: void smatreqdev, exit(2222); \n"); exit(2222);

       printf(" m->location = %d, LOCDEV = %d\n", m->location, LOCDEV);
       printf("   m->row_ptr   = %p\n", (void *) m->row_ptr);  
       printf("   m->col_idx   = %p\n", (void *) m->col_idx);  
       printf("   m->values    = %p\n", (void *) m->values);  
       printf("   m->idxColMem = %p\n", (void *) m->idxColMem);  
       printf("   m->extra     = %p\n", (void *) m->extra);  
    if (m->location != LOCDEV) {
        m->location = LOCDEV;
        m->idxColMem = m->col_idx;  // Transfer idx_col to idxColMem
        
        //int* arr = (int*)m->idxColMem;  // Cast to the expected type (e.g., int*)
        //printf("em smatreqdev, :: idxColMem = ");
        //for (int i = 0; i < m->nnz; i++) { printf("%d ", ((int*) m->idxColMem)[i]); } printf("\n");
        
        m->extra   = m->values;  // Transfer values to extra
        //m->col_idx = NULL;  // Clear idx_col
        //m->values  = NULL;  // Clear values
    }
       printf("em smatreqdev, ANTES DO fim\n");
       printf(" m->location  = %d, LOCDEV = %d\n", m->location, LOCDEV);
       printf(" nao coloque NULL.  m->col_idx   = %p\n", (void *) m->col_idx);  
       printf("   m->idxColMem = %p\n", (void *) m->idxColMem);  
       printf(" nao coloque NULL.  m->values = %p\n", (void *) m->values);  
       printf("   m->extra  = %p\n", (void *) m->extra);  
//     printf("em smatreqdev, fim\n"); exit(2222);
}

void smatrix_delete(smatrix_t *smatrix) {
    printf("BD, ATENCAO, em %s: void smatrix_delete(smatrix_t *smatrix ), NO FREE! {\n", __FILE__); // _NAME__);
    return;
    if (!smatrix) { return; }
    // Free linked lists in smat if allocated
    /*
    if (smatrix->smat) {
        for (int i = 0; i < smatrix->nrow; i++) {
            if (smatrix->smat[i]) {
                slist_clear(smatrix->smat[i]);  // Clear the list contents
                free(smatrix->smat[i]);         // Ensure the slist struct itself is freed
                smatrix->smat[i] = NULL;
            }
        }
        free(smatrix->smat);  // Free the array of slist pointers
        smatrix->smat = NULL;

         printf("BD, ATENCAO, em smatrix_delete, REMOVIDO free(smatrix->idxColMem);\n");
         //free(smatrix->idxColMem);
         smatrix->idxColMem = NULL;
    }
    */
    // Free CSR-related arrays if allocated
    //  
    free(smatrix->row_ptr);
    smatrix->row_ptr = NULL;

    free(smatrix->col_idx);
    smatrix->col_idx = NULL;

    free(smatrix->values);
    smatrix->values = NULL;

    // Free additional pointers if allocated
    printf("BD, ATENCAO, em smatrix_delete, REMOVIDO  free(smatrix->extra);\n");
    //if (  smatrix->extra != NULL ) free(smatrix->extra);
    smatrix->extra = NULL;

    // Finally, free the struct itself
    free(smatrix);
    printf("BD, em smatrix_delete, FINAL ;\n");
}

#ifndef __FILE_NAME__
#define __FILE_NAME__ __FILE__
#endif


#include <math.h>
void print_vectorT(vector_t *v_) {
    if (v_ == NULL) { printf("BD, em %s: print_vectorT, vetor NULL\n", __FILE__); return; }

    int n = v_->len;
    if (n <= 0) { printf("BD, em %s: print_vectorT, vetor vazio\n", __FILE__); return; }

    //if (v_->extra == NULL) { printf("BD, em %s: print_vectorT, v_->extra é NULL\n", __FILE__); return; }

    printf("BD, em %s: print_vectorT, ", __FILE_NAME__); setvbuf(stdout, NULL, _IONBF, 0);

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


void print_smatrix(const smatrix_t* matrix) {
    printf("em %s: void print_smatrix(const smatrix_t* matrix)\n",__FILE__);
    if (!matrix) {
        printf("Matrix is NULL.\n");
        return;
    }

    printf("Matrix (%p):\n", (void*)matrix);
    printf("  Rows: %d, Cols: %d, NNZ: %d\n", matrix->nrow, matrix->ncol, matrix->nnz);
    printf("  isPacked: %d\n", matrix->isPacked);
    printf("  type: %d\n", matrix->type);
    printf("  location: %u\n", matrix->location);
    printf("  extra: %p\n", matrix->extra);
    printf("  idxColMem: %p\n", matrix->idxColMem);

    if (matrix->row_ptr) {
        printf("  row_ptr: ");
        for (int i = 0; i <= matrix->nrow; i++) {
            printf("%lld ", matrix->row_ptr[i]);
        }
        printf("\n");
    } else {
        printf("  row_ptr is NULL.\n");
    }

    if (matrix->col_idx) {
        printf("  col_idx: ");
        for (int i = 0; i < matrix->nnz; i++) {
            printf("%lld ", matrix->col_idx[i]);
        }
        printf("\n");
    } else {
        printf("  col_idx is NULL.\n");
    }

    if (matrix->values) {
        printf("  values: ");
        for (int i = 0; i < matrix->nnz; i++) {
            printf("%.4f ", matrix->values[i]);
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

