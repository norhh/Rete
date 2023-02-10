#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <dirent.h>
#include <klee/klee.h>
#include <stdbool.h>

int table_miss = 1;

struct entry_s {
    char *key;
    int value;
    struct entry_s *next;
};

typedef struct entry_s entry_t;

struct hashtable_s {
    int size;
    struct entry_s **table;
};

typedef struct hashtable_s hashtable_t;


/* Create a new hashtable. */
hashtable_t *ht_create( int size ) {

    hashtable_t *hashtable = NULL;
    int i;

    if( size < 1 ) return NULL;

    /* Allocate the table itself. */
    if( ( hashtable = malloc( sizeof( hashtable_t ) ) ) == NULL ) {
        return NULL;
    }

    /* Allocate pointers to the head nodes. */
    if( ( hashtable->table = malloc( sizeof( entry_t * ) * size ) ) == NULL ) {
        return NULL;
    }
    for( i = 0; i < size; i++ ) {
        hashtable->table[i] = NULL;
    }

    hashtable->size = size;

    return hashtable;
}

/* Hash a string for a particular hash table. */
int ht_hash( hashtable_t *hashtable, char *key ) {

    unsigned long int hashval;
    int i = 0;

    /* Convert our string to an integer */
    while( hashval < ULONG_MAX && i < strlen( key ) ) {
        hashval = hashval << 8;
        hashval += key[ i ];
        i++;
    }

    return hashval % hashtable->size;
}

/* Create a key-value pair. */
entry_t *ht_newpair( char *key, int value ) {
    entry_t *newpair;

    if( ( newpair = malloc( sizeof( entry_t ) ) ) == NULL ) {
        return NULL;
    }

    if( ( newpair->key = strdup( key ) ) == NULL ) {
        return NULL;
    }

    newpair->value = value;

    newpair->next = NULL;

    return newpair;
}

/* Insert a key-value pair into a hash table. */
void ht_set( hashtable_t *hashtable, char *key, int value) {
    int bin = 0;
    entry_t *newpair = NULL;
    entry_t *next = NULL;
    entry_t *last = NULL;

    bin = ht_hash( hashtable, key );

    next = hashtable->table[ bin ];

    while( next != NULL && next->key != NULL && strcmp( key, next->key ) > 0 ) {
        last = next;
        next = next->next;
    }

    /* There's already a pair.  Let's replace that string. */
    if( next != NULL && next->key != NULL && strcmp( key, next->key ) == 0 ) {

        next->value = value;

        /* Nope, could't find it.  Time to grow a pair. */
    } else {
        newpair = ht_newpair( key, value );

        /* We're at the start of the linked list in this bin. */
        if( next == hashtable->table[ bin ] ) {
            newpair->next = next;
            hashtable->table[ bin ] = newpair;

            /* We're at the end of the linked list in this bin. */
        } else if ( next == NULL ) {
            last->next = newpair;

            /* We're in the middle of the list. */
        } else  {
            newpair->next = next;
            last->next = newpair;
        }
    }
}

/* Retrieve a key-value pair from a hash table. */
int ht_get( hashtable_t *hashtable, char *key ) {
    int bin = 0;
    entry_t *pair;

    bin = ht_hash( hashtable, key );

    /* Step through the bin, looking for our value. */
    pair = hashtable->table[ bin ];
    while( pair != NULL && pair->key != NULL && strcmp( key, pair->key ) > 0 ) {
        pair = pair->next;
    }

    /* Did we actually find anything? */
    if( pair == NULL || pair->key == NULL || strcmp( key, pair->key ) != 0 ) {
        table_miss = 1;
        return 0;

    } else {
        table_miss = 0;
        return pair->value;
    }

}

/*
  End of hashtable implementation
 */

#define MAX_NAME_LENGTH 1000

hashtable_t *choice_instances;
hashtable_t *output_instances;
hashtable_t *selector_instances;
hashtable_t *output_var_instances;

void init_tables() {
    choice_instances = ht_create(65536);
    output_instances = ht_create(65536);
    selector_instances = ht_create(65536);
    output_var_instances = ht_create(65536);
}

int __trident_choice(char* lid,
                     int** rvals_int, bool** rvals_bool, char** rvals_ids_int, char** rvals_ids_bool,
                     int rvals_size_int, int rvals_size_bool,
                     int** lvals_int, bool** lvals_bool, char** lvals_ids_int, char** lvals_ids_bool,
                     int lvals_size_int, int lvals_size_bool){
    if (!choice_instances)
        init_tables();

    int previous = ht_get(choice_instances, lid);
    int instance;
    if (table_miss) {
        instance = 0;
    } else {
        instance = previous + 1;
    }
    ht_set(choice_instances, lid, instance);

    for (int i=0; i<rvals_size_int; i++) {
        char name[MAX_NAME_LENGTH];
        sprintf(name, "choice!rvalue!%s!%d!%s", lid, instance, rvals_ids_int[i]);
        int klee_var;
        klee_make_symbolic(&klee_var, sizeof(klee_var), name);
        klee_assume(klee_var == *rvals_int[i]);
    }
    for (int i=0; i<rvals_size_bool; i++) {
        char name[MAX_NAME_LENGTH];
        sprintf(name, "choice!rvalue!%s!%d!%s", lid, instance, rvals_ids_bool[i]);
        int klee_var;
        klee_make_symbolic(&klee_var, sizeof(klee_var), name);
        klee_assume(klee_var == *rvals_bool[i]);
    }

    bool condition = true;
    char selector_name[MAX_NAME_LENGTH];
    int selectors_int[lvals_size_int];
    int selectors_bool[lvals_size_bool];
    for(int i=0;i<lvals_size_int; i++) {
        sprintf(selector_name, "choice!lvalue!selector!%s!%s", lid, lvals_ids_int[i]);
        int selector = ht_get(selector_instances, selector_name);
        if (table_miss) {
            klee_make_symbolic(&selector, sizeof(selector), selector_name);
            klee_assume(selector == 1 | selector == 0);
            ht_set(selector_instances, selector_name, selector);
        }
        selectors_int[i] = selector;
    }
    for(int i=0;i<lvals_size_bool; i++) {
        sprintf(selector_name, "choice!lvalue!selector!%s!%s", lid, lvals_ids_bool[i]);
        int selector = ht_get(selector_instances, selector_name);
        if (table_miss) {
            klee_make_symbolic(&selector, sizeof(selector), selector_name);
            ht_set(selector_instances, selector_name, selector);
            klee_assume(selector == 1 | selector == 0);
        }
        selectors_bool[i] = selector;
    }

    int selector_sum = 0;

    for (int i = 0; i < lvals_size_int; i++) {
        char next_name[MAX_NAME_LENGTH];
        int selector = selectors_int[i];
        klee_assume(selector == 1 | selector == 0);
        sprintf(next_name, "choice!lvalue!%s!%d!%s", lid, instance, lvals_ids_int[i]);
        int next_klee_var;
        klee_make_symbolic(&next_klee_var, sizeof(next_klee_var), next_name);
        int prev_lvalue = *lvals_int[i];
        *lvals_int[i] = next_klee_var;
        condition = condition & ((prev_lvalue == next_klee_var) | (selector == 1));
        selector_sum += selector;
    }
    for (int i = 0; i < lvals_size_bool; i++) {
        char next_name[MAX_NAME_LENGTH];
        int selector = selectors_bool[i];
        klee_assume(selector == 1 | selector == 0);
        sprintf(next_name, "choice!lvalue!%s!%d!%s", lid, instance, lvals_ids_bool[i]);
        int next_klee_var;
        klee_make_symbolic(&next_klee_var, sizeof(next_klee_var), next_name);
        int prev_lvalue = *lvals_bool[i];
        *lvals_bool[i] = next_klee_var;
        condition = condition & ((prev_lvalue == next_klee_var) | (selector == 1));
        condition  = condition & (next_klee_var == 1 | next_klee_var ==  0);
        selector_sum += selector;
    }

    klee_assume(condition & (selector_sum <= 3));
    char name[MAX_NAME_LENGTH];
    sprintf(name, "choice!angelic!i32!%s!%d", lid, instance);
    int angelic;
    klee_make_symbolic(&angelic, sizeof(angelic), name);
    return angelic;
}


int __trident_choice_semfix(char* lid,
                     int** rvals_int, bool** rvals_bool, char** rvals_ids_int, char** rvals_ids_bool,
                     int rvals_size_int, int rvals_size_bool,
                     int** lvals_int, bool** lvals_bool, char** lvals_ids_int, char** lvals_ids_bool,
                     int lvals_size_int, int lvals_size_bool){
    if (!choice_instances)
        init_tables();

    int previous = ht_get(choice_instances, lid);
    int instance;
    if (table_miss) {
        instance = 0;
    } else {
        instance = previous + 1;
    }
    ht_set(choice_instances, lid, instance);

    for (int i=0; i<rvals_size_int; i++) {
        char name[MAX_NAME_LENGTH];
        sprintf(name, "choice!rvalue!%s!%d!%s", lid, instance, rvals_ids_int[i]);
        int klee_var;
        klee_make_symbolic(&klee_var, sizeof(klee_var), name);
        klee_assume(klee_var == *rvals_int[i]);
    }
    for (int i=0; i<rvals_size_bool; i++) {
        char name[MAX_NAME_LENGTH];
        sprintf(name, "choice!rvalue!%s!%d!%s", lid, instance, rvals_ids_bool[i]);
        int klee_var;
        klee_make_symbolic(&klee_var, sizeof(klee_var), name);
        klee_assume(klee_var == *rvals_bool[i]);
    }
    klee_open_merge();
    bool condition = true;
    char selector_name[MAX_NAME_LENGTH];
    int selectors_int[lvals_size_int];
    int selectors_bool[lvals_size_bool];
    for(int i=0;i<lvals_size_int; i++) {
        sprintf(selector_name, "choice!lvalue!selector!%s!%s", lid, lvals_ids_int[i]);
        int selector = ht_get(selector_instances, selector_name);
        if (table_miss) {
            klee_make_symbolic(&selector, sizeof(selector), selector_name);
            ht_set(selector_instances, selector_name, selector);
            klee_assume(selector == 1 | selector == 0);
        }
        selectors_int[i] = selector;
    }
    for(int i=0;i<lvals_size_bool; i++) {
        sprintf(selector_name, "choice!lvalue!selector!%s!%s", lid, lvals_ids_bool[i]);
        int selector = ht_get(selector_instances, selector_name);
        if (table_miss) {
            klee_make_symbolic(&selector, sizeof(selector), selector_name);
            ht_set(selector_instances, selector_name, selector);
            klee_assume(selector == 1 | selector == 0);
        }
        selectors_bool[i] = selector;
    }
    int selector_sum=0;
    for (int i=0; i<lvals_size_int; i++) {
        char next_name[MAX_NAME_LENGTH];
        sprintf(next_name, "choice!lvalue!%s!%d!%s", lid, instance, lvals_ids_int[i]);
        int var;
        klee_make_symbolic(&var, sizeof(var), next_name);
        int prev_lvalue = *lvals_int[i];
        *lvals_int[i] = var;
        if(selectors_int[i] == 0)
            klee_assume(prev_lvalue == var);
        selector_sum+=selectors_int[i];
    }
    for (int i=0; i<lvals_size_bool; i++) {
        char next_name[MAX_NAME_LENGTH];
        sprintf(next_name, "choice!lvalue!%s!%d!%s", lid, instance, lvals_ids_bool[i]);
        int var;
        klee_make_symbolic(&var, sizeof(var), next_name);
        klee_assume(var== 0 | var==1);

        int prev_lvalue = *lvals_bool[i];
        *lvals_bool[i] = var;
        if(selectors_bool[i] == 0)
            klee_assume(prev_lvalue == var);
        selector_sum+=selectors_bool[i];
    }

    klee_close_merge();
    //klee_assume(selector_sum<3);
    char name[MAX_NAME_LENGTH];
    sprintf(name, "choice!angelic!i32!%s!%d", lid, instance);
    int angelic;
    klee_make_symbolic(&angelic, sizeof(angelic), name);

    return angelic;
}
int __trident_choice_bool_semfix(char* lid,
                            bool** rvals, char** rvals_ids, int rvals_size,
                            bool** lvals, char** lvals_ids, int lvals_size){
    if (!choice_instances)
        init_tables();
    int previous = ht_get(choice_instances, lid);
    int instance;
    if (table_miss) {
        instance = 0;
    } else {
        instance = previous + 1;
    }
    ht_set(choice_instances, lid, instance);
    for (int i=0; i<rvals_size; i++) {
        char name[MAX_NAME_LENGTH];
        sprintf(name, "choice!rvalue!%s!%d!%s", lid, instance, rvals_ids[i]);
        int klee_var;
        klee_make_symbolic(&klee_var, sizeof(klee_var), name);
        klee_assume(klee_var == *rvals[i]);
    }
    char selector_name[MAX_NAME_LENGTH];
    int selectors[lvals_size];
    for(int i=0;i<lvals_size; i++) {
        sprintf(selector_name, "choice!lvalue!selector!%s!%s", lid, lvals_ids[i]);
        int selector = ht_get(selector_instances, selector_name);
        if (table_miss) {
            klee_make_symbolic(&selector, sizeof(selector), selector_name);
            ht_set(selector_instances, selector_name, selector);
        }
        selectors[i] = selector;
        klee_assume(selector == 0 | selector == 1);
    }
    int next_klee_var[lvals_size+1];
    for(int i=0; i<lvals_size; i++) {
        char next_name[MAX_NAME_LENGTH];
        sprintf(next_name, "choice!lvalue!%s!%d!%s", lid, instance, lvals_ids[i]);
        int var;
        klee_make_symbolic(&var, sizeof(var), next_name);
        next_klee_var[i] = var;
        klee_assume(var== 0 | var==1);
    }
    klee_open_merge();
    int selector_sum=0;
    for (int i=0; i<lvals_size; i++) {
        int prev_lvalue = *lvals[i];
        *lvals[i] = next_klee_var[i];
        if(selectors[i] == 0)
            klee_assume(prev_lvalue == next_klee_var[i]);
        selector_sum+=selectors[i];
    }
    klee_close_merge();
    //klee_assume(selector_sum<3);
    char name[MAX_NAME_LENGTH];
    sprintf(name, "choice!angelic!i32!%s!%d", lid, instance);
    int angelic;
    klee_make_symbolic(&angelic, sizeof(angelic), name);

    return angelic;
}

int __trident_output(char* id, char* typestr, int value){
    if (!output_instances)
        init_tables();
    int previous = ht_get(output_instances, id);
    int instance;
    if (table_miss) {
      instance = 0;
    } else {
      instance = previous + 1;
    }
    ht_set(output_instances, id, instance);
    char name[MAX_NAME_LENGTH];
    sprintf(name, "output!%s!%s!%d", typestr, id, instance);
    int klee_var = ht_get(output_var_instances, name);
    if (table_miss) {
        klee_make_symbolic(&klee_var, sizeof(klee_var), name);
        ht_set(output_var_instances, name, klee_var);
    }

    klee_assume(klee_var == value);
    return value;
}