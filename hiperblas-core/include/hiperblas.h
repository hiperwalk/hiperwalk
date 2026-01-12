#ifndef __HIPERBLAS_H__
#define __HIPERBLAS_H__

#ifdef	__cplusplus
extern "C" {
#endif

#define SYM_NO_ERROR             0x000000
#define SYM_ALREADY_IN_TABLE     0x000001
#define SYM_MEM_FAIL             0x000002

#define FUNC_NO_ERROR            0x100000
#define FUNC_ALREADY_IN_TABLE    0x100001
#define FUNC_INVALID_PARAMS      0x000001
#define NEB_PRESENT "Neblina Parallel Runtime Interpreter %s, (C) LNCC 2010-2015.\n", 1
#define LOCHOS 1
#define LOCDEV 128

typedef struct __func_info_t {
    char name[256];                           // Name
    int nparams;                              // Number of params of a function
    int nreturns;                             // Number of return of a function
    int * type_params;
    int * type_returns;    
    void ** (*ptr_func)( void ** input, int * status );
} func_info_t;

enum mem_avail_type { 
                      MEM_AVAIL, 
                      MEM_ALLOCATED 
                    };
typedef enum  {
        T_STRING,
        T_INT,
        T_FLOAT,
        T_COMPLEX,
        T_ADDR,
        T_NDEF,
        T_LIST,
        T_STRTOREL,
        T_CFUNC,
        T_VECTOR,
        T_MATRIX,
        T_SMATRIX,
        T_RMATRIX,
        T_FILE,
        T_ANY
} data_type;

typedef enum  {
        F_NATIVE,
        F_CANSI
} func_type;

typedef union __data_u {
        char *       s;
        int          i;
        double       f;
        unsigned int a;
        void *       v;
        int  *       e;
 func_info_t *       p; 
} data_u;

typedef struct __symbol_t {
    char              name[256];
    data_type         type;
    int               position;
} symbol_t;

typedef struct __function_t {
    char              name[256];
    int               nparams;
    int               rparams;
    int               init_addr;
    int               final_addr;
    func_type         type;
    int               min_store; 
    int               max_store;
} function_t;


typedef struct __object_t {
    data_type         type;
    data_u            value;
} object_t;

typedef union __data_vector_u {
    void               * v;
    int                * i;
    double              * f;
    void              ** s;
} data_vector_u;

typedef struct __slist {
    int col;
    double re;
    double im;
    struct __slist * next;

} slist;

//typedef struct __rmatrix_t {
//    int nrow;
//    int ncol;
//    cl_kernel kernel;
//    int id;
//}rmatrix_t;    
    

typedef struct __symbol_table_t {
    symbol_t                * value;
    struct __symbol_table_t * next;
} symbol_table_t;


typedef struct __scope_table_t {
    symbol_table_t         * table;
    struct __scope_table_t * next;
} scope_table_t;


typedef struct __function_table_t {
    scope_table_t            * scope;
    function_t               * func;
    struct __function_table_t   * next;
} function_table_t;


enum code_ops {  STORE, JMP_FALSE, GOTO, CALL, CALL_C, CALL_THREAD, STORE_REF, ALLOCV, LOADV, STOREV,
                DATA, LOAD, LD_INT, LD_VAR, INC, NEQ, ANDL, ORL, ANDA, ORA, XOR, NEG, NEGL, SHL, SHR, RELMEM,
                READ_INT, WRITE_INT, READ_FLOAT, READ_STRING, APPEND, ALLOCS, SIGCH, MOD, DEC, GOTO_ST, LDADDR,
                LT, EQ, GT, LEQ, GEQ, ADD, SUB, MUL, DIV, PWR, PUSH, 
                PUSH_PC, POP, WRITE, HALT 
              };

typedef struct __instruction {

    enum code_ops operation;
    object_t      arg;   
        
} instruction_t;

typedef struct __info_source {

    int           line;     // curr_line
    int           file;     // In include_hist 
        
} info_source_t;


typedef struct __std_list {
    object_t obj;
    struct __std_list * next;
} list_t;



#define type(v)    ((v).type)
#define ivalue(v)  ((v).value.i)
#define fvalue(v)  ((v).value.f)
#define avalue(v)  ((v).value.a)
#define svalue(v)  ((v).value.s)
#define pvalue(v)  ((v).value.p)
#define vvalue(c)  ((c).value.v)
#define evalue(v)  ((v).value.e)


#ifdef	__cplusplus
}
#endif

#endif
