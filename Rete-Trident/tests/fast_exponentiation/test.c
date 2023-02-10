#include <stdio.h>

#ifndef TRIDENT_OUTPUT
#define TRIDENT_OUTPUT(id, typestr, value) value
#endif

long long int MD = 1000000000 + 7;

long long power(int a, int k) {
    long long p;

    if (k == 0)
        return 1;
    p = power(a, k / 2);
    p = p * p % MD;
    if (k % 2)
        p = p * a % MD;
    return p;
}

long long power_proxy(int *a, int k) {
    *a = power(*a, k);
}

int main(int argc, char *argv[]) {
    /*
    int a, k, a1;
    klee_make_symbolic(&a, sizeof(a), "value_arg1_a");
    klee_make_symbolic(&k, sizeof(k), "value_arg2_k");
    klee_make_symbolic(&a1, sizeof(a1), "value_arg1_a_DONE");

    power_proxy(&a, k);
    klee_assume(a==a1);
    */
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
    int x=atoi(argv[1]), b=atoi(argv[2]);
    __trident_choice("9", (int*[]){&x, &b},0, (char*[]){"x", "b"},0, 2,0, (int*[]){&x},0, (char*[]){"x"},0, 1, 0);
    return TRIDENT_OUTPUT("x", "i32", x);
}
