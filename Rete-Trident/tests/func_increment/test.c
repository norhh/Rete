#include <stdio.h>

#ifndef TRIDENT_OUTPUT
#define TRIDENT_OUTPUT(id, typestr, value) value
#endif

void crazy_function(int* a, int b) {
    if(b>10)
        *a += b;
    else
        *a -= b;
}
void increment(int *a){
    (*a)++;
}
int main(int argc, char *argv[]) {
  int x=atoi(argv[1]);
  int a2, b2;
  /*
  klee_make_symbolic(&a, sizeof(a), "value_a");
  klee_make_symbolic(&b, sizeof(b), "value_b");
  crazy_function(&a, b);
  klee_make_symbolic(&a2, sizeof(a2), "value_a_DONE");
  klee_make_symbolic(&b2, sizeof(b2), "value_b_DONE");

  klee_assume(a != a2 | b!= b2);
  */
  /*klee_make_symbolic(&a, sizeof(a), "value_a");
  klee_make_symbolic(&b, sizeof(a), "value_a_DONE");
  increment(&a);
  klee_assume(a!=b);
  */
  __trident_choice("9", (int*[]){&x},0, (char*[]){"x"},0, 1,0, (int*[]){&x},0, (char*[]){"x"},0, 1,0);
  return TRIDENT_OUTPUT("x", "i32", x);
}
