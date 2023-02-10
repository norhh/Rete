#include <stdio.h>

#ifndef TRIDENT_OUTPUT
#define TRIDENT_OUTPUT(id, typestr, value) value
#endif

//standard convex hull code from https://www.geeksforgeeks.org/convex-hull-set-1-jarviss-algorithm-or-wrapping/
#include <stdio.h>
#include <stdlib.h>


int gcd(int a, int b)
{
    if (a == 0)
        return b;
    return gcd(b % a, a);
}
int get_gcd(int a, int b, int *ans) {
    *ans = gcd(a, b);
}
int main(int argc, char *argv[]) {
    int a=atoi(argv[1]), b=atoi(argv[2]), x=0;


    __trident_choice("9", (int*[]){&a, &b, &x},0,
            (char*[]){"a", "b", "x"},0, 3,0, (int*[]){&x},0, (char*[]){"x"},0, 1,0);

    /*
    int a, b, ans, ans_new;

    klee_make_symbolic(&a, sizeof(a), "value_arg1_a");
    klee_make_symbolic(&b, sizeof(b), "value_arg2_b");
    klee_make_symbolic(&ans, sizeof(ans), "value_arg3_ans");
    klee_make_symbolic(&ans_new, sizeof(ans_new), "value_arg3_ans_new");

    get_gcd(a, b, &ans);
    klee_assume(ans == ans_new);
    */
    /*klee_make_symbolic(&a, sizeof(a), "value_a");
    klee_make_symbolic(&b, sizeof(a), "value_a_DONE");
    increment(&a);
    klee_assume(a!=b);
    */
    //int x=atoi(argv[1]), b=atoi(argv[2]);
    //__trident_choice("L9", "i32", (int*[]){&x, &b}, (char*[]){"x", "b"}, 2, (int*[]){&x}, (char*[]){"x"}, 1);
    return TRIDENT_OUTPUT("x", "i32", x);
}
