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
int main(int argc, char *argv[]) {

    /*
  int a,b,a2, b2;
  klee_make_symbolic(&a, sizeof(a), "value_arg1_a");
  klee_make_symbolic(&b, sizeof(b), "value_arg2_b");
  crazy_function(&a, b);
  klee_make_symbolic(&a2, sizeof(a2), "value_arg1_a_DONE");
  klee_make_symbolic(&b2, sizeof(b2), "value_arg2_b_DONE");

  klee_assume(a == a2);
    */
  /*
  klee_make_symbolic(&a, sizeof(a), "value_a");
  klee_make_symbolic(&b, sizeof(a), "value_a_DONE");
  increment(&a);
  klee_assume(a!=b);
  */

    int x=atoi(argv[1]), b=atoi(argv[2]);
    __trident_choice("L9", "i32", (int*[]){&x, &b}, (char*[]){"x", "b"}, 2, (int*[]){&x}, (char*[]){"x"}, 1);
  return TRIDENT_OUTPUT("x", "i32", x);

}
