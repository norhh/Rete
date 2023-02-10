//
// Created by nikhil on 12/11/2019.
//
#include<stdbool.h>
#ifndef TRIDENT_RUNTIME_H
#define TRIDENT_RUNTIME_H


#define TRIDENT_OUTPUT(id, typestr, value) \
  __trident_output(id, typestr, value)

int __trident_choice(char* lid,
                     int** rvals_int, bool** rvals_bool, char** rvals_ids_int, char** rvals_ids_bool,
                     int rvals_size_int, int rvals_size_bool,
                     int** lvals_int, bool** lvals_bool, char** lvals_ids_int, char** lvals_ids_bool,
                     int lvals_size_int, int lvals_size_bool);

int __trident_choice_semfix(char* lid,
                     int** rvals_int, bool** rvals_bool, char** rvals_ids_int, char** rvals_ids_bool,
                     int rvals_size_int, int rvals_size_bool,
                     int** lvals_int, bool** lvals_bool, char** lvals_ids_int, char** lvals_ids_bool,
                     int lvals_size_int, int lvals_size_bool);


int __trident_choice_bool_semfix(char* lid,
                          bool** rvals, char** rvals_ids, int rvals_size,
                          bool** lvals, char** lvals_ids, int lvals_size);

int __trident_choice_int_semfix(char* lid,
                     int** rvals, char** rvals_ids, int rvals_size,
                     int** lvals, char** lvals_ids, int lvals_size);

int __trident_output(char* id, char* typestr, int value);


#endif //TRIDENT_RUNTIME_H
