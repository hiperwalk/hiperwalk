#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <string.h>
#include "Python.h"
#include <numpy/arrayobject.h>
#include "hiperblas.h"
#include "hiperblas_std.h"
#include "hiperblas_vector.h"
#include "hiperblas_matrix.h"
#include "hiperblas_smatrix.h"
#include "hiperblas_complex.h"
#include "bridge_api.h"
#include "libhiperblas.h"


bridge_manager_t bridge_manager;
int bridge_index = 0;

static PyObject* py_init_engine(PyObject* self, PyObject* args){
    //cl_int err;
    //cl_uint num_platforms;
    int device;
    int bridge;
    
    if (!PyArg_ParseTuple(args, "ii", &bridge, &device)) return NULL;
    //err = clGetPlatformIDs(0, NULL, &num_platforms);
    //if (err == CL_SUCCESS) {
            //std::cout << "Success. Platforms available: " << num_platforms
            //        << std::endl;
    //} else {
            //std::cout << "Error. Platforms available: " << num_platforms
            //        << std::endl;
    //}

    //InitCLEngine(device);
    const char *home = getenv("HOME");
    bridge_index = device;
    char  *lib_name; 
    lib_name = (char*) malloc ( 1024 *sizeof(char));
    switch(bridge){
        case 0:
/*
            lib_name = "/usr/local/lib64/libhiperblas-cpu-bridge.so";
            lib_name = "/home/bidu/libs/lib/libhiperblas-cpu-bridge.so";
            lib_name = "~/hiperblas/lib/libhiperblas-cpu-bridge.so";
            lib_name = "/prj/prjedlg/bidu/hiperblas/lib/libhiperblas-cpu-bridge.so";
            lib_name = "/home/bidu/hiperblas/lib/libhiperblas-cpu-bridge.so";
	    */
	    snprintf(lib_name, 1024, "%s%s", home,"/hiperblas/lib/libhiperblas-cpu-bridge.so");
            printf("lib_name = %s\n", lib_name);
            break;
        case 1:
	    snprintf(lib_name, 1024, "%s%s", home,"/hiperblas/lib64/libhiperblas-opencl-bridge.so");
            break;
        default:
	    snprintf(lib_name, 1024, "%s%s", home,"/hiperblas/lib64/libhiperblas-cpu-bridge.so");
            break;
    }
    printf("BD, em ./pyhiperblas/hiperblas_wrapper.c: static PyObject* py_init_engine, lib_name =%s\n", lib_name); //exit(2222);
														   
    load_plugin(&bridge_manager, lib_name, bridge_index);
    bridge_manager.bridges[bridge_index].InitEngine_f(device);
    // printf("3\n");
    setvbuf(stdout, NULL, _IONBF, 0); // BD
    Py_RETURN_NONE;
}

static PyObject* py_stop_engine(PyObject* self, PyObject* args){
    //ReleaseCLInfo(clinfo);
    release_plugin(&bridge_manager, bridge_index);
//    double v1[2];
//    double v2[2];
//    double *res;
//    v1[0] = 1;
//    v1[1] = 2;
//    v2[0] = 1;
//    v2[1] = 2;
//    res = bridge_manager.bridges[bridge_index].addVectorF_f(&v1,&v2,2);
//    for (int i=0;i<2;i++) {
//        printf("%f\n",res[i]);
//    }
    Py_RETURN_NONE;
}

static void py_complex_delete(PyObject* self) {
    complex_t* comp = (complex_t*)PyCapsule_GetPointer(self, "py_complex_new");
    bridge_manager.bridges[bridge_index].complex_delete(comp);
    //Py_RETURN_NONE;
}


static PyObject* py_complex_new(PyObject* self, PyObject* args){
    double real;
    double imag;
    if (!PyArg_ParseTuple(args, "dd", &real, &imag)) return NULL;

    complex_t * a = bridge_manager.bridges[bridge_index].complex_new(real, imag);

    PyObject* po = PyCapsule_New((void*)a, "py_complex_new", py_complex_delete);

    return po;
}

static void py_vector_delete(PyObject *capsule) 
{
    vector_t *vec = (vector_t*) PyCapsule_GetPointer(capsule, "py_vector_new");
    if (vec == NULL) return;

    // Chama seu destrutor da biblioteca C
    bridge_manager.bridges[bridge_index].vector_delete(vec);
}



static PyObject* py_vector_new(PyObject* self, PyObject* args){
    int len;
    int data_type;
    if (!PyArg_ParseTuple(args, "ii", &len,&data_type)) return NULL;
    //printf("create %d\n",len);
    vector_t * a = bridge_manager.bridges[bridge_index].vector_new(len, data_type, 1, NULL);
    //printf("malloc %p\n",a);
    PyObject* po = PyCapsule_New((void*)a, "py_vector_new", py_vector_delete);
    //printf("capsule_new %p\n",po);
    return po;
}

static PyObject* py_load_numpy_array(PyObject* self, PyObject* args){
    PyObject* a = NULL;

    if(!PyArg_ParseTuple(args, "O:py_load_numpy_array", &a)) return NULL;

    float* dataArrayA = (float*)PyArray_DATA((PyArrayObject*)a);

    npy_intp* shape = PyArray_DIMS((PyArrayObject*)a);

    int len = shape[0];
    PyArray_Descr* dtype = PyArray_DESCR((PyArrayObject*) a);
    
    int data_type = (dtype->kind == 'f' ? T_FLOAT : T_COMPLEX);

    vector_t * vec_a = bridge_manager.bridges[bridge_index].vector_new(len, data_type, 0, dataArrayA);

    PyObject* po = PyCapsule_New((void*)vec_a, "py_vector_new", py_vector_delete);

    return po;
}

static PyObject* py_retrieve_numpy_array(PyObject* self, PyObject* args){
    
    PyObject* pf = NULL;

    if(!PyArg_ParseTuple(args, "O:py_retrieve_numpy_array", &pf)) return NULL;

    // printf("pf %p\n",pf);
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    // printf("vec %p\n",vec);
    // printf("%p\n",vec->value.f);
    bridge_manager.bridges[bridge_index].vecreqhost(vec);
    vec->externalData = 1; //to make sure it will not be deallocated

    // for (int i=0; i < vec->len; i++){
    //     printf("%d %lf\n", i, vec->value.f[i]);
    // }
    int rows = vec->len;
    npy_intp dims[1] = {rows};  // Array dimensions
    int data_type = (vec->type == T_FLOAT ? NPY_FLOAT64 : NPY_COMPLEX128);

    PyObject* numpyArray = PyArray_SimpleNewFromData(1, dims, data_type, vec->value.f);

    // double* outputData = (double*)PyArray_DATA((PyArrayObject*)numpyArray);

    // for (int i=0; i < dims[0]; i++) {
    //     printf("%d %f\n",i,outputData[i]);
    // }

    // Make sure to set flags to the NumPy array to manage memory correctly
    PyArray_ENABLEFLAGS((PyArrayObject*)numpyArray, NPY_ARRAY_OWNDATA);
    
    return numpyArray;
}

static PyObject* py_vector_set(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int n;
    double real;
    double imag;
    if(!PyArg_ParseTuple(args, "Oidd:py_vector_set", &pf, &n, &real, &imag)) return NULL;

    //printf("print %d\n",n);
    //printf("pf %p\n",pf);
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    //printf("vec %p\n",vec);
    if (vec->type == T_COMPLEX) {
        vec->value.f[2*n] = real;
        vec->value.f[2*n+1] = imag;
    } else {
        vec->value.f[n] = real;
    }
    //printf("%lf\n",vec->value.f[n]);
    Py_RETURN_NONE;
}

static PyObject* py_vector_get(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int n;
    if(!PyArg_ParseTuple(args, "Oi:py_vector_get", &pf, &n)) return NULL;

    // printf("print %d\n",n);
    // printf("pf %p\n",pf);
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    // printf("vec %p\n",vec);
    // printf("%lf\n",vec->value.f[n]);
    PyObject * result = PyFloat_FromDouble((double)vec->value.f[n]);
    return result;
}

static PyObject* py_move_vector_device(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_vector_device", &pf)) return NULL;

    //printf("pf %p\n",pf);
    
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    //printf("vec %p\n",vec);
    bridge_manager.bridges[bridge_index].vecreqdev(vec);
    Py_RETURN_NONE;
}

static PyObject* py_copy_vector_from_device(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_vector_device", &pf)) return NULL;

    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    object_t ** in = convertToObject(vec_a, NULL);

    vector_t * r = (vector_t *) copy_vector_from_device(&bridge_manager, bridge_index, (void **) in, NULL );

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;

}

static PyObject* py_move_vector_host(PyObject* self, PyObject* args) {


    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_vector_host", &pf)) return NULL;

    //printf("pf %p\n",pf);
    
    vector_t * vec = (vector_t *)PyCapsule_GetPointer(pf, "py_vector_new");
    //printf("vec %p\n",vec);
    bridge_manager.bridges[bridge_index].vecreqhost(vec);
    Py_RETURN_NONE;
    //vecreqdev(vec);
    //PyObject* po = PyCapsule_New((void*)vec, "py_vector_new", py_vector_delete);
    //return po;

}
//
static PyObject* py_vec_add(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_vec_add", &a, &b)) return NULL;

    //printf("a %p\n",a);
    //printf("b %p\n",b);
    
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
        //printf("vec_a %p\n",vec_a);
    vector_t * vec_b = (vector_t *)PyCapsule_GetPointer(b, "py_vector_new");
        //printf("vec_b %p\n",vec_b);

    
    //TODO completar o vec_add
    object_t ** in = convertToObject(vec_a,vec_b);
    
    //printf("vec add to call\n");
    vector_t * r = (vector_t *) vec_add(&bridge_manager, bridge_index, (void **) in, NULL );
    
    //TODO this part returns a numpy_array (not working yet)
    // PyObject* outputArray = PyArray_SimpleNew(2, shape, data_type);
    // printf("%d\n",i++);

    // float* outputData = (float*)PyArray_DATA((PyArrayObject*)outputArray);
    // printf("%d\n",i++);

    // printf("%p %p\n",outputData,r->value.f);
    // for (int i=0; i < r->len; i++) {
    //     printf("%d %f\n",i,r->value.f[i]);
    // }
    // memcpy(outputData, r->value.f, r->len);
    // printf("%d\n",i++);

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_matvec_mul(PyObject* self, PyObject* args) {
    printf("BD, em ./pyhiperblas/hiperblas_wrapper.c: static PyObject* py_matvec_mul(PyObject* self, ... \n");
    PyObject* M = NULL; PyObject* vI = NULL; PyObject* vO = NULL;
    if(!PyArg_ParseTuple(args, "OOO:py_matvec_mul", &M, &vI, &vO)) return NULL;
    matrix_t * mat_M = (matrix_t *)PyCapsule_GetPointer(M,  "py_matrix_new");
    vector_t * vec_I = (vector_t *)PyCapsule_GetPointer(vI, "py_vector_new");
    vector_t * vec_O = (vector_t *)PyCapsule_GetPointer(vO, "py_vector_new");
    object_t ** in = convertToObject3BD(mat_M, vec_I, vec_O);
    matvec_mul3BD(&bridge_manager, bridge_index, (void **) in, NULL );
//vector_t * r = (vector_t *) matvec_mul3(&bridge_manager, bridge_index, (void **) in, NULL );
    Py_RETURN_NONE;
}

// Destrutor do PyCapsule
static void py_sparse_matrix_delete(PyObject* capsule) {
    printf("BD, ATENCAO !!!  em ./pyhiperblas/hiperblas_wrapper.c: static void py_sparse_matrix_delete\n");
    smatrix_t* a = (smatrix_t*) PyCapsule_GetPointer(capsule, "py_sparse_matrix_new");
    //free (a->values); free (a);
    //exit(128+37);
    //return;
    if (a) {
        // Chame a função apropriada do seu backend para liberar memória
        bridge_manager.bridges[bridge_index].smatrix_delete(a);
    }
    //Py_RETURN_NONE;
}

static PyObject* py_permute_sparse_matrix(PyObject* self, PyObject* args) {
    printf("BD, em ./pyhiperblas/hiperblas_wrapper.c: static PyObject* py_permute_sparse_matrix(PyObject* self, ... \n");
    PyObject* objS = NULL; PyObject* objC = NULL; PyObject* objU = NULL;
    if(!PyArg_ParseTuple(args, "OOO:py_permute_sparse_matrix", &objS, &objC, &objU)) return NULL;
    smatrix_t * hb_smatS = (smatrix_t *) PyCapsule_GetPointer(objS, "py_sparse_matrix_new");
    smatrix_t * hb_smatC = (smatrix_t *) PyCapsule_GetPointer(objC, "py_sparse_matrix_new");
    smatrix_t * hb_smatU = (smatrix_t *) PyCapsule_GetPointer(objU, "py_sparse_matrix_new");
    permuteSparseMatrix(&bridge_manager, bridge_index, hb_smatS, hb_smatC, hb_smatU );
    Py_RETURN_NONE;
}

static PyObject* py_sparse_matvec_mul(PyObject* self, PyObject* args) {
    printf("BD, em ./pyhiperblas/hiperblas_wrapper.c: static PyObject* py_sparse_matvec_mul(PyObject* self, ... \n");

    PyObject* m = NULL; PyObject* vIn = NULL;  PyObject* vOut = NULL;
    if(!PyArg_ParseTuple(args, "OOO:py_sparse_matvec_mul", &m, &vIn, &vOut)) return NULL;

    smatrix_t * aSmat   = (smatrix_t *)PyCapsule_GetPointer(m,    "py_sparse_matrix_new");
    vector_t  * vec_In  = (vector_t  *)PyCapsule_GetPointer(vIn,  "py_vector_new");
    vector_t  * vec_Out = (vector_t  *)PyCapsule_GetPointer(vOut, "py_vector_new");

    object_t ** in      = convertToObject4BD(aSmat, vec_In, vec_Out );
    matvec_mul3BD(&bridge_manager, bridge_index, (void **) in, NULL );

    Py_RETURN_NONE;
}

static PyObject* py_vec_prod(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_vec_add", &a, &b)) return NULL;

    //printf("a %p\n",a);
    //printf("b %p\n",b);
    
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
        //printf("vec_a %p\n",vec_a);
    vector_t * vec_b = (vector_t *)PyCapsule_GetPointer(b, "py_vector_new");
        //printf("vec_b %p\n",vec_b);

    
    object_t ** in = convertToObject(vec_a,vec_b);
    
    vector_t * r = (vector_t *) vec_prod(&bridge_manager, bridge_index, (void **) in, NULL );
    

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_vec_add_off(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    int offset;
    if(!PyArg_ParseTuple(args, "iO:py_vec_add_off", &offset, &a)) return NULL;

    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");

    object_t ** in = convertToObject2(offset, vec_a);
    
    vector_t * r = (vector_t *) vec_add_off(&bridge_manager, bridge_index, (void **) in, NULL );   

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_vec_sum(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    if(!PyArg_ParseTuple(args, "O:py_vec_sum", &a)) return NULL;

    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
    
    object_t ** in = convertToObject(vec_a, NULL);
    
    object_t * r = (object_t *) vec_sum(&bridge_manager, bridge_index, (void **) in, NULL );

    PyObject * result = PyFloat_FromDouble((double)r->value.f);
    
    return result;
}

static PyObject* py_vec_conj(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    if(!PyArg_ParseTuple(args, "O:py_vec_sum", &a)) return NULL;

    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
    
    object_t ** in = convertToObject(vec_a, NULL);
    
    vector_t * r = (vector_t *) vec_conj(&bridge_manager, bridge_index, (void **) in, NULL );

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

//

static void py_matrix_delete(PyObject* self) {
    matrix_t* mat = (matrix_t*)PyCapsule_GetPointer(self, "py_matrix_new");
    //printf("mat %p\n",mat);
    //printf("mat->value %p\n",&(mat->value));
    //free ((void *)mat->value.f);
    //free ((void *)mat);
    bridge_manager.bridges[bridge_index].matrix_delete(mat);
    //Py_RETURN_NONE;
}
/*
static void py_smatrix_delete(PyObject* self) {
    smatrix_t* mat = (smatrix_t*)PyCapsule_GetPointer(self, "py_smatrix_new");
    //printf("mat %p\n",mat);
    //printf("mat->value %p\n",&(mat->value));
    //free ((void *)mat->value.f);
    //free ((void *)mat);
    bridge_manager.bridges[bridge_index].smatrix_delete(mat);
    //Py_RETURN_NONE;
}
*/

/*
    // Limpa referências temporárias
    Py_DECREF(dataArr);
    Py_DECREF(indicesArr);
    Py_DECREF(indptrArr);
    Py_DECREF(dataObj);
    Py_DECREF(indicesObj);
    Py_DECREF(indptrObj);

*/
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* py_smatrix_connect(PyObject* self, PyObject* args)
{
    printf("BD, em %s: static PyObject* py_smatrix_connect( ...\n", __FILE__);// , __func__)

    // Argumentos: (capsule, csr_matrix)
    PyObject *aSmatObj = NULL, *csr_obj = NULL;
    if (!PyArg_ParseTuple(args, "OO:py_smatrix_connect", &aSmatObj, &csr_obj))
        return NULL;

    // Recupera ponteiro da cápsula
    smatrix_t* smat_a = (smatrix_t*) PyCapsule_GetPointer(aSmatObj, "py_sparse_matrix_new");
    if (!smat_a) {
        PyErr_SetString(PyExc_RuntimeError, "PyCapsule inválida para smatrix_t");
        return NULL;
    }
    int data_type = smat_a->type ;

    // Atributos da CSR
    PyObject *dataObj    = PyObject_GetAttrString(csr_obj, "data");
    PyObject *indicesObj = PyObject_GetAttrString(csr_obj, "indices");
    PyObject *indptrObj  = PyObject_GetAttrString(csr_obj, "indptr");

    if (!dataObj || !indicesObj || !indptrObj ) {
        PyErr_SetString(PyExc_TypeError, "Objeto não é csr_matrix válido");
        return NULL;
    }

    // --- Verifica tipos e contiguidade (sem cópias) ---
    PyArrayObject *indptrArr  = (PyArrayObject*) PyArray_FROM_OTF(indptrObj,  NPY_INT64, NPY_ARRAY_CARRAY);
    PyArrayObject *indicesArr = (PyArrayObject*) PyArray_FROM_OTF(indicesObj, NPY_INT64, NPY_ARRAY_CARRAY);
    PyArrayObject *dataArr=NULL;
    if(data_type==T_FLOAT   ) {dataArr    = (PyArrayObject*) PyArray_FROM_OTF(dataObj, NPY_FLOAT64,    NPY_ARRAY_CARRAY);}
    if(data_type==T_COMPLEX ) {dataArr    = (PyArrayObject*) PyArray_FROM_OTF(dataObj, NPY_COMPLEX128, NPY_ARRAY_CARRAY);}


    if (!PyArray_Check(indptrArr) || PyArray_TYPE(indptrArr) != NPY_INT64 || !PyArray_ISCARRAY(indptrArr)) {
        PyErr_SetString(PyExc_TypeError, "indptr deve ser np.int64 contíguo"); return NULL;
    }
    if (!PyArray_Check(indicesArr) || PyArray_TYPE(indicesArr) != NPY_INT64 || !PyArray_ISCARRAY(indicesArr)) {
        PyErr_SetString(PyExc_TypeError, "indices deve ser np.int64 contíguo"); return NULL;
    }

    if (data_type == T_COMPLEX || data_type == T_FLOAT) {
        int expected_type = (data_type == T_COMPLEX) ? NPY_COMPLEX128 : NPY_FLOAT64;
        const char *type_str = (data_type == T_COMPLEX) ? "np.complex128 contíguo" : "np.float64 contíguo";

        if (!PyArray_Check(dataArr) ||
            PyArray_TYPE((PyArrayObject *)dataArr) != expected_type ||
            !PyArray_ISCARRAY((PyArrayObject *)dataArr)) {
                PyErr_Format(PyExc_TypeError, "data deve ser %s", type_str);
                return NULL;
            }
     }

    // --- Compartilhamento real de memória ---
    smat_a->row_ptr = (npy_int64*)   PyArray_DATA(indptrArr);
    smat_a->col_idx = (npy_int64*)   PyArray_DATA(indicesArr);
    if(data_type == T_FLOAT)   smat_a->values  = (npy_float64*) PyArray_DATA(dataArr);
    if(data_type == T_COMPLEX) smat_a->values  = (npy_complex128*) PyArray_DATA(dataArr);

    smat_a->nnz  = smat_a->row_ptr[smat_a->nrow];
    smat_a->isPacked = 1;

    printf("BD2, CSR spipy conectado com smatrix HB: nrow=%ld, ncol=%ld, nnz=%ld\n",
           (long int) smat_a->nrow, (long int) smat_a->ncol, (long int) smat_a->nnz);

    //print_smatrix(smat_a); 

    Py_RETURN_NONE;
}

static PyObject* py_load_numpy_matrix(PyObject* self, PyObject* args){
    PyObject* a = NULL;

    if(!PyArg_ParseTuple(args, "O:py_load_numpy_matrix", &a)) return NULL;

    float* dataArrayA = (float*)PyArray_DATA((PyArrayObject*)a);

    npy_intp* shape = PyArray_DIMS((PyArrayObject*)a);

    int rows = shape[0];
    int cols = shape[1];
    PyArray_Descr* dtype = PyArray_DESCR((PyArrayObject*) a);

    int data_type = (dtype->kind == 'f' ? T_FLOAT : T_COMPLEX);

    matrix_t * mat_a = bridge_manager.bridges[bridge_index].matrix_new(rows, cols, data_type, 0, dataArrayA);

    PyObject* po = PyCapsule_New((void*)mat_a, "py_matrix_new", py_matrix_delete);

    return po;
}

static PyObject* py_retrieve_numpy_matrix(PyObject* self, PyObject* args){
    
    PyObject* pf = NULL;

    if(!PyArg_ParseTuple(args, "O:py_retrieve_numpy_matrix", &pf)) return NULL;

    // printf("pf %p\n",pf);
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    // printf("vec %p\n",vec);
    // printf("%p\n",vec->value.f);
    bridge_manager.bridges[bridge_index].matreqhost(mat);
    mat->externalData = 1; //to make sure it will not be deallocated

    // for (int i=0; i < vec->len; i++){
    //     printf("%d %lf\n", i, vec->value.f[i]);
    // }
    int rows = mat->nrow;
    int cols = mat->ncol;
    npy_intp dims[2] = {rows, cols};  // Array dimensions
    int data_type = (mat->type == T_FLOAT ? NPY_FLOAT64 : NPY_COMPLEX128);

    PyObject* numpyArray = PyArray_SimpleNewFromData(2, dims, data_type, mat->value.f);

    // double* outputData = (double*)PyArray_DATA((PyArrayObject*)numpyArray);

    // for (int i=0; i < dims[0]; i++) {
    //     printf("%d %f\n",i,outputData[i]);
    // }

    // Make sure to set flags to the NumPy array to manage memory correctly
    PyArray_ENABLEFLAGS((PyArrayObject*)numpyArray, NPY_ARRAY_OWNDATA);
    
    return numpyArray;
}

static PyObject* py_matrix_new(PyObject* self, PyObject* args){
    int rows;
    int cols;
    int data_type;
    if (!PyArg_ParseTuple(args, "iii", &rows,&cols,&data_type)) return NULL;
    //printf("create %d\n",rows);
    //printf("create %d\n",cols);
    matrix_t * a = bridge_manager.bridges[bridge_index].matrix_new(rows, cols, data_type, 1, NULL);
    //printf("malloc %p\n",a);
    PyObject* po = PyCapsule_New((void*)a, "py_matrix_new", py_matrix_delete);
    //printf("capsule_new %p\n",po);
    return po;
}

static PyObject* py_matrix_set(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int i;
    int j;
    double real;
    double imag;
    if(!PyArg_ParseTuple(args, "Oiidd:py_matrix_set", &pf, &i, &j, &real, &imag)) return NULL;

    //printf("print (%d,%d)\n",i,j);
    //printf("pf %p\n",pf);
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    //printf("mat %p\n",mat);
    int idx = (i*mat->ncol + j);
    if (mat->type == T_COMPLEX) {
        mat->value.f[2 * idx] = real;
        mat->value.f[2 * idx + 1] = imag;
    } else {
        mat->value.f[idx] = real;
    }
    //printf("%lf\n",mat->value.f[i*mat->ncol + j]);
    Py_RETURN_NONE;
}

static PyObject* py_matrix_get(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int i;
    int j;
    if(!PyArg_ParseTuple(args, "Oii:py_matrix_get", &pf, &i, &j)) return NULL;

    //printf("print (%d,%d)\n",i,j);
    //printf("pf %p\n",pf);
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    //printf("mat %p\n",mat);
    //printf("%lf\n",mat->value.f[i*mat->ncol + j]);
    PyObject * result = PyFloat_FromDouble((double)mat->value.f[i*mat->ncol + j]);
    return result;
}


static PyObject* py_move_matrix_device(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_matrix_device", &pf)) return NULL;

    //printf("pf %p\n",pf);
    
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    //printf("mat %p\n",mat);
    bridge_manager.bridges[bridge_index].matreqdev(mat);
    Py_RETURN_NONE;
}

static PyObject* py_move_matrix_host(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_matrix_device", &pf)) return NULL;

    //printf("pf %p\n",pf);
    
    matrix_t * mat = (matrix_t *)PyCapsule_GetPointer(pf, "py_matrix_new");
    bridge_manager.bridges[bridge_index].matreqhost(mat); //should use this function? It seems that it creates the object in the stack
    //    //printf("mat %p\n",mat);
//    int n = (mat->type==T_FLOAT?mat->nrow*mat->ncol:2*mat->nrow*mat->ncol);
//    matrix_t * out = matrix_new(mat->nrow, mat->ncol, mat->type);
//    cl_int status = clEnqueueReadBuffer(clinfo.q, mat->extra, CL_TRUE, 0, n * sizeof (double), out->value.f, 0, NULL, NULL);
//    CLERR
//    PyObject* po = PyCapsule_New((void*)out, "py_matrix_new", py_matrix_delete);
//    return po;
    Py_RETURN_NONE;
}



#define SMATRIX_CAPSULE_NAME "hiperblas.smatrix"


/*
static PyObject* BD00py_sparse_matrix_new(PyObject* self, PyObject* args){
    int rows, cols, data_type;
    if (!PyArg_ParseTuple(args, "iii", &rows, &cols, &data_type)) return NULL;
    smatrix_t *a = bridge_manager.bridges[bridge_index].smatrix_new(rows, cols, data_type);
    if (!a) {
        PyErr_SetString(PyExc_RuntimeError, "smatrix_new failed"); return NULL;
    }
    PyObject* po = PyCapsule_New((void*)a, SMATRIX_CAPSULE_NAME, py_sparse_matrix_delete);
    return po;
}
*/

static PyObject* py_sparse_matrix_new(PyObject* self, PyObject* args){
    int rows; int cols; int data_type;
    printf("BD, em pyhiperblas/hiperblas_wrapper.c: py_sparse_matrix_new( PyObject* self, PyObject* args) \n");
    if (!PyArg_ParseTuple(args, "iii", &rows, &cols, &data_type)) return NULL;
    smatrix_t * a = bridge_manager.bridges[bridge_index].smatrix_new(rows, cols, data_type);
    if (!a) { PyErr_SetString(PyExc_RuntimeError, "smatrix_new failed"); return NULL; }
    PyObject* po = PyCapsule_New((void*)a, "py_sparse_matrix_new", py_sparse_matrix_delete);
    if (!po) {
        // evita vazamento caso PyCapsule_New falhe
        bridge_manager.bridges[bridge_index].smatrix_delete(a); return NULL;
    }
    return po;
}


/*
static PyObject* py_sparse_matrix_new(PyObject* self, PyObject* args){
    int rows; int cols; int data_type;
    printf("BD, em pyhiperblas/hiperblas_wrapper.c: py_sparse_matrix_new( PyObject* self, PyObject* args) \n");
    if (!PyArg_ParseTuple(args, "O", &scipyCsrMat)) return NULL;
    rows=scipyCsrMat.shape[0]
    // if (!PyArg_ParseTuple(args, "OOO:", &)) return NULL;
    printf(" call bridge_manager.bridges[bridge_index].smatrix_new(rows, cols, data_type);\n");
    smatrix_t * a = bridge_manager.bridges[bridge_index].smatrix_new(rows, cols, data_type);
    //printf("em py_sparse_matrix_new, a.value[0]=%f, ", a->values[0]); //printf("exit(2223);\n"); exit(2223);
    if (!a) { PyErr_SetString(PyExc_RuntimeError, "smatrix_new failed"); return NULL; }
    printf(" before CALL PyObject* po = PyCapsule_New((void*)a, py_sparse_matrix_new, py_sparse_matrix_delete\n");
    PyObject* po = PyCapsule_New((void*)a, "py_sparse_matrix_new", py_sparse_matrix_delete);
    // printf(" after  CALL PyObject* po = PyCapsule_New((void*)a, py_sparse_matrix_new, py_sparse_matrix_delete\n");
    if (!po) {
        // evita vazamento caso PyCapsule_New falhe
        bridge_manager.bridges[bridge_index].smatrix_delete(a); return NULL;
    }
    printf("em py_sparse_matrix_new FINAL %p\n",po);
    //printf("em py_sparse_matrix_new, " ); printf("exit(2223);\n"); exit(2223);
    return po;
}

*/

static PyObject* py_print_vectorT      (PyObject* self, PyObject* args) {
    PyObject* pV = NULL;
    printf("BD, em hiperblas_wraper.c: static PyObject* py_print_vectorT(PyObject* self, PyObject* args)\n");
    if(!PyArg_ParseTuple(args, "O:py_print_vectorT", &pV)) {printf("return NULL\n"); exit(2222);  return NULL;}
    vector_t * v  = (vector_t *)PyCapsule_GetPointer(pV, "py_vector_new");
    //print_vectorT(v); 
    //exit(2222);
    //     OU    OU   a linha de baixo 
    bridge_manager.bridges[bridge_index].print_vectorT_f(v);
    Py_RETURN_NONE;
}
static PyObject* py_sparse_matrix_print(PyObject* self, PyObject* args) {
    PyObject* pM = NULL;
    printf("BD, em hiperblas_wraper.c: static PyObject* py_sparse_matrix_print(PyObject* self, PyObject* args)\n");
    if(!PyArg_ParseTuple(args, "O:py_sparse_matrix_print", &pM)) return NULL;
    smatrix_t * sMat = (smatrix_t *)PyCapsule_GetPointer(pM, "py_sparse_matrix_new");
    printf("BD, em hiperblas_wraper.c: static PyObject* py_sparse_matrix_print(.., CALL print_smatrix(sMat); \n");
    print_smatrix(sMat); 
    //     OU    OU   a linha de baixo 
    //printf("BD, em hiperblas_wraper.c: static PyObject* py_sparse_matrix_print(.., CALL bridge_manager.bridges[bridge_index].print_smatrix_f(sMat); \n");
    //bridge_manager.bridges[bridge_index].print_smatrix_f(sMat);
    Py_RETURN_NONE;
}

static PyObject* py_sparse_matrix_set(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    int i;
    int j;
    double real;
    double imag;
    if(!PyArg_ParseTuple(args, "Oiidd:py_sparse_matrix_set", &pf, &i, &j, &real, &imag)) return NULL;

    //printf("print (%d,%d)\n",i,j);
    //printf("pf %p\n",pf);
    smatrix_t * mat = (smatrix_t *)PyCapsule_GetPointer(pf, "py_sparse_matrix_new");
    //printf("smat %p\n",mat);
    if(mat->type == T_COMPLEX) {
        bridge_manager.bridges[bridge_index].smatrix_set_complex_value(mat,i,j,real, imag);
    } else if(mat->type == T_FLOAT) {
        bridge_manager.bridges[bridge_index].smatrix_set_real_value(mat,i,j,real);
    }
    Py_RETURN_NONE;
}

static PyObject* py_sparse_matrix_pack(PyObject* self, PyObject* args) {
    
    printf("BD, em %s: static PyObject* py_sparse_matrix_Pack(PyObject* self,\n", __FILE__);// , __func__)
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_sparse_matrix_pack", &pf)) return NULL;

    //printf("pf %p\n",pf);
    smatrix_t * mat = (smatrix_t *)PyCapsule_GetPointer(pf, "py_sparse_matrix_new");
    if (mat->type == T_FLOAT) {
        bridge_manager.bridges[bridge_index].smatrix_pack(mat);
    } else {
        bridge_manager.bridges[bridge_index].smatrix_pack_complex(mat);
    }
    printf("BD, em py_sparse_matrix_Pack( ... , FINAL\n");// , __func__)
    Py_RETURN_NONE;
}

static PyObject* py_move_sparse_matrix_device(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_sparse_matrix_device", &pf)) return NULL;

    smatrix_t * smat = (smatrix_t *)PyCapsule_GetPointer(pf, "py_sparse_matrix_new");
    bridge_manager.bridges[bridge_index].smatreqdev(smat);
    Py_RETURN_NONE;
}

static PyObject* py_move_sparse_matrix_host(PyObject* self, PyObject* args) {
    
    PyObject* pf = NULL;
    if(!PyArg_ParseTuple(args, "O:py_move_sparse_matrix_device", &pf)) return NULL;

    smatrix_t * smat = (smatrix_t *)PyCapsule_GetPointer(pf, "py_sparse_matrix_new");
    bridge_manager.bridges[bridge_index].smatreqhost(smat); //should use this function? It seems that it creates the object in the stack
    //printf("mat %p\n",mat);
//    int n = (mat->type==T_FLOAT?mat->nrow*mat->ncol:2*mat->nrow*mat->ncol);
//    matrix_t * out = matrix_new(mat->nrow, mat->ncol, mat->type);
//    cl_int status = clEnqueueReadBuffer(clinfo.q, mat->mem, CL_TRUE, 0, n * sizeof (double), out->value.f, 0, NULL, NULL);
//    CLERR
//    PyObject* po = PyCapsule_New((void*)smat, "py_sparse_matrix_new", py_sparse_matrix_delete);
//    return po;
    Py_RETURN_NONE;
}

static PyObject* py_mat_add(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_mat_add", &a, &b)) return NULL;

    //printf("a %p\n",a);
    //printf("b %p\n",b);
    

    matrix_t * mat_a = (matrix_t *)PyCapsule_GetPointer(a, "py_matrix_new");
        //printf("vec_a %p\n",vec_a);
    matrix_t * mat_b = (matrix_t *)PyCapsule_GetPointer(b, "py_matrix_new");
        //printf("vec_b %p\n",vec_b);

    
    //TODO completar o vec_add
    object_t ** in = convertMatMatToObject(mat_a,mat_b);
    
    matrix_t * r = (matrix_t *) mat_add(&bridge_manager, bridge_index, (void **) in, NULL );
    

    PyObject* po = PyCapsule_New((void*)r, "py_matrix_new", py_matrix_delete);
    return po;
}

static PyObject* py_mat_mul(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    PyObject* b = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_mat_mul", &a, &b)) return NULL;

    //printf("a %p\n",a);
    //printf("b %p\n",b);
    
    matrix_t * mat_a = (matrix_t *)PyCapsule_GetPointer(a, "py_matrix_new");
        //printf("vec_a %p\n",vec_a);
    matrix_t * mat_b = (matrix_t *)PyCapsule_GetPointer(b, "py_matrix_new");
        //printf("vec_b %p\n",vec_b);

    
    //TODO completar o vec_add
    object_t ** in = convertMatMatToObject(mat_a,mat_b);
    
    matrix_t * r = (matrix_t *) mat_mul(&bridge_manager, bridge_index, (void **) in, NULL );
    

    PyObject* po = PyCapsule_New((void*)r, "py_matrix_new", py_matrix_delete);
    return po;
}

static PyObject* py_scalar_mat_mul(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    double scalar;
    if(!PyArg_ParseTuple(args, "dO:py_scalar_mat_mul", &scalar,&a)) return NULL;

    matrix_t * mat_a = (matrix_t *)PyCapsule_GetPointer(a, "py_matrix_new");
    
    object_t ** in = convertScaMatToObject(scalar, mat_a);
    
    matrix_t * r = (matrix_t *) mat_mulsc(&bridge_manager, bridge_index, (void **) in, NULL );

    PyObject* po = PyCapsule_New((void*)r, "py_matrix_new", py_matrix_delete);
    return po;
}

static PyObject* py_scalar_vec_mul(PyObject* self, PyObject* args) {
    
    PyObject* a = NULL;
    double scalar;
    if(!PyArg_ParseTuple(args, "dO:py_scalar_vec_mul", &scalar,&a)) return NULL;

    //printf("a %p\n",a);
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
    //printf("vec_a %p\n",vec_a);
    
    object_t ** in = convertScaVecToObject(scalar, vec_a);
    
    vector_t * r = (vector_t *) vec_mulsc(&bridge_manager, bridge_index, (void **) in, NULL );
    //    printf("r %p\n",r);
    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_complex_scalar_vec_mul(PyObject* self, PyObject* args) {
    
    PyObject* scalar = NULL;
    PyObject* a = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_complex_scalar_vec_mul", &scalar,&a)) return NULL;

    complex_t * complex_scalar = (complex_t *)PyCapsule_GetPointer(scalar, "py_complex_new");
    vector_t * vec_a = (vector_t *)PyCapsule_GetPointer(a, "py_vector_new");
    
    vector_t * r = NULL;
    if (vec_a->type == T_FLOAT) {
        r = (vector_t *) vec_mul_complex_scalar (&bridge_manager, bridge_index,  complex_scalar, vec_a); 
    } else if (vec_a->type == T_COMPLEX) {
        r = (vector_t *) mul_complex_scalar_complex_vec(&bridge_manager, bridge_index,  complex_scalar, vec_a);
    }

    PyObject* po = PyCapsule_New((void*)r, "py_vector_new", py_vector_delete);
    return po;
}

static PyObject* py_complex_scalar_mat_mul(PyObject* self, PyObject* args) {
    
    PyObject* scalar = NULL;
    PyObject* a = NULL;
    if(!PyArg_ParseTuple(args, "OO:py_complex_scalar_mat_mul", &scalar,&a)) return NULL;

    complex_t * complex_scalar = (complex_t *)PyCapsule_GetPointer(scalar, "py_complex_new");
    matrix_t * mat_a = (matrix_t *)PyCapsule_GetPointer(a, "py_matrix_new");
    
    matrix_t * r = NULL;
    if (mat_a->type == T_FLOAT) {
        r = (matrix_t *) mul_complex_scalar_float_mat (&bridge_manager, bridge_index,  complex_scalar, mat_a); 
    } else if (mat_a->type == T_COMPLEX) {
        r = (matrix_t *) mul_complex_scalar_complex_mat(&bridge_manager, bridge_index,  complex_scalar, mat_a);
    }

    PyObject* po = PyCapsule_New((void*)r, "py_matrix_new", py_matrix_delete);
    return po;
}

static PyObject* cpu_constant;
static PyObject* gpu_constant;
static PyObject* float_constant;
static PyObject* complex_constant;

static PyObject* get_cpu_constant(PyObject* self, PyObject* args)
{
    Py_INCREF(cpu_constant);
    return cpu_constant;
}

static PyObject* get_gpu_constant(PyObject* self, PyObject* args)
{
    Py_INCREF(gpu_constant);
    return gpu_constant;
}

static PyObject* get_float_constant(PyObject* self, PyObject* args)
{
    Py_INCREF(float_constant);
    return float_constant;
}

static PyObject* get_complex_constant(PyObject* self, PyObject* args)
{
    Py_INCREF(complex_constant);
    return complex_constant;
}

static PyMethodDef mainMethods[] = {
    {"init_engine", py_init_engine, METH_VARARGS, "init_engine"},
    {"stop_engine", py_stop_engine, METH_VARARGS, "stop_engine"},
    {"vector_new",  py_vector_new, METH_VARARGS, "vector_new"},
    {"load_numpy_array", py_load_numpy_array, METH_VARARGS, "load_numpy_array"},
    {"retrieve_numpy_array", py_retrieve_numpy_array, METH_VARARGS, "retrieve_numpy_array"},
    {"vector_set", py_vector_set, METH_VARARGS, "vector_set"},
    {"vector_get", py_vector_get, METH_VARARGS, "vector_get"},
//{"print_vectorT", py_print_vectorT, METH_VARARGS, "escrever na tela os elementos do vetor de reais do tipo vector_t"},
    {"move_vector_device", py_move_vector_device, METH_VARARGS, "move_vector_device"},
    {"copy_vector_from_device", py_copy_vector_from_device, METH_VARARGS, "copy_vector_from_device"},
    {"move_vector_host", py_move_vector_host, METH_VARARGS, "move_vector_host"},

    {"matrix_new", py_matrix_new, METH_VARARGS, "matrix_new"},
    {"load_numpy_matrix", py_load_numpy_matrix, METH_VARARGS, "load_numpy_matrix"},
    {"retrieve_numpy_matrix", py_retrieve_numpy_matrix, METH_VARARGS, "retrieve_numpy_matrix"},
    {"matrix_set", py_matrix_set, METH_VARARGS, "matrix_set"},
    {"matrix_get", py_matrix_get, METH_VARARGS, "matrix_get"},
    {"move_matrix_device", py_move_matrix_device, METH_VARARGS, "move_matrix_device"},
    {"move_matrix_host", py_move_matrix_host, METH_VARARGS, "move_matrix_host"},
    {"sparse_matrix_new", py_sparse_matrix_new, METH_VARARGS, "sparse_matrix_new"},
    //{"sparse_new", py_matrix_new, METH_VARARGS, "sparse_new"},
    {"smatrix_connect", py_smatrix_connect, METH_VARARGS, "ponteiros de iCSR scipy connect at smatrix hiperblas struct"},
    {"sparse_matrix_set", py_sparse_matrix_set, METH_VARARGS, "sparse_matrix_set"},
    {"sparse_matrix_pack", py_sparse_matrix_pack, METH_VARARGS, "sparse_matrix_pack"},
    {"move_sparse_matrix_device", py_move_sparse_matrix_device, METH_VARARGS, "move_sparse_matrix_device"},
    {"move_sparse_matrix_host", py_move_sparse_matrix_host, METH_VARARGS, "move_sparse_matrix_host"},
    {"sparse_matvec_mul", py_sparse_matvec_mul, METH_VARARGS, "sparse_matvec_mul por bidu"},
    {"sparse_matrix_print", py_sparse_matrix_print, METH_VARARGS, "sparse_matrix_print"}, //BD
    {"print_vectorT",       py_print_vectorT,       METH_VARARGS, "print_vectorT"},
    {"vec_add", py_vec_add, METH_VARARGS, "vec_add"},
    {"matvec_mul", py_matvec_mul, METH_VARARGS, "matvec_mul"},
    {"permute_sparse_matrix", py_permute_sparse_matrix, METH_VARARGS, "permute_sparse_matrix"},
    {"vec_prod", py_vec_prod, METH_VARARGS, "vec_prod"},
    {"vec_add_off", py_vec_add_off, METH_VARARGS, "vec_add_off"},
    {"vec_sum", py_vec_sum, METH_VARARGS, "vec_sum"},
    {"vec_conj", py_vec_conj, METH_VARARGS, "vec_conj"},
    {"mat_add", py_mat_add, METH_VARARGS, "mat_add"},
    {"mat_mul", py_mat_mul, METH_VARARGS, "mat_mul"},
    {"scalar_mat_mul", py_scalar_mat_mul, METH_VARARGS, "scalar_mat_mul"},
    {"scalar_vec_mul", py_scalar_vec_mul, METH_VARARGS, "scalar_vec_mul"},
    {"complex_scalar_vec_mul", py_complex_scalar_vec_mul, METH_VARARGS, "complex_scalar_vec_mul"},
    {"complex_scalar_mat_mul", py_complex_scalar_mat_mul, METH_VARARGS, "complex_scalar_mat_mul"},
    {"complex_new", py_complex_new, METH_VARARGS, "complex_new"},
    {"get_cpu_constant", get_cpu_constant, METH_NOARGS, "Get the CPU constant value."},
    {"get_gpu_constant", get_gpu_constant, METH_NOARGS, "Get the GPU constant value."},
    {"get_float_constant", get_float_constant, METH_NOARGS, "Get the FLOAT constant value."},
    {"get_complex_constant", get_complex_constant, METH_NOARGS, "Get COMPLEX the constant value."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef hiperblas = {
    PyModuleDef_HEAD_INIT,
    "hiperblas", "Hiperblas Core",
    -1,
    mainMethods
};

PyMODINIT_FUNC PyInit_hiperblas(void) {
    PyObject* module = PyModule_Create(&hiperblas);
    cpu_constant     = PyLong_FromLong(0);
    gpu_constant     = PyLong_FromLong(1);
    float_constant   = PyLong_FromLong(2);
    complex_constant = PyLong_FromLong(3);
    
    PyModule_AddObject(module, "CPU",     cpu_constant);
    PyModule_AddObject(module, "GPU",     gpu_constant);
    PyModule_AddObject(module, "FLOAT",   float_constant);
    PyModule_AddObject(module, "COMPLEX", complex_constant);
    import_array();
    return module;
}


