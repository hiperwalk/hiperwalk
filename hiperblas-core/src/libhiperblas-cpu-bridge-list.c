#include "hiperblas_list.h"
#include <stdio.h>
#include <stdlib.h>

list_t * list_new() {
    return (list_t *) NULL;
}

list_t * list_append( list_t * L, object_t o ) {
    list_t * newnode = (list_t *) malloc( sizeof(list_t) );
    newnode->obj = o;
    newnode->next = NULL;
        
    if( L == NULL ) {
        return newnode;
    } else {    
        list_t * ptr = L;
        while( ptr->next != NULL )
            ptr = ptr->next;
        ptr->next = newnode;
        return L;
    }
}

int list_len( list_t * L ) {
   list_t * ptr = L;
   int ret = 0;
   while( ptr != NULL ) {
        ret++;
        ptr = ptr->next; 
   }
   return ret;
} 

object_t list_get( list_t * L, int i) {
    list_t * ptr = L;
    int ret = 1;
    while( ptr != NULL ) {
        if( ret == i )
            return ptr->obj; 
        ret++;
        ptr = ptr->next; 
    }
    fprintf(stderr, "Runtime error: list get() index out of range\n" );
    exit( 1 );
}
