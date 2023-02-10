#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef TRIDENT_OUTPUT
#define TRIDENT_OUTPUT(id, typestr, value) value
#endif

int main(int argc, char** argv) {
  int x = atoi(argv[1]);
  int y = 6;
  int len = 7;
  int arr[len];
  int* arr_save[len];
  char* arr_save_names[len];
  for(int i=0; i<len; i++) {
    arr[i] = (i+1)*35;
    arr_save[i] = &arr[i];
    arr_save_names[i] = malloc(sizeof(char)*50);
    sprintf(arr_save_names[i], "a[%d]", i);
  }
  __trident_choice("12", arr_save, (bool*[]){}, arr_save_names ,(char*[]){} , len, 0,  arr_save, (bool*[]){}, arr_save_names ,(char*[]){} , len, 0);
  
  return TRIDENT_OUTPUT("x", "i32", x);
}
