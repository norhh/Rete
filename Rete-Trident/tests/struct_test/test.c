#include <stdio.h>

#ifndef TRIDENT_OUTPUT
#define TRIDENT_OUTPUT(id, typestr, value) value
#endif

long long int MD = 1000000000 + 7;

struct something {
 int a,b;
 char name;
};

void struct_manipulator(struct something* a, int c) {
    a->a += c;
    a->b -= c;
    a->name+=1;
}

int main(int argc, char *argv[]) {

    struct something a, b;
    int c;
    klee_make_symbolic(&c, sizeof(c), "value_arg2_c");
    klee_make_symbolic(&a, sizeof(a), "value_arg1_a");
    klee_make_symbolic(&b, sizeof(b), "value_arg1_a_DONE");
    struct_manipulator(&a, c);
    klee_assume(a.a == b.a & a.b == b.b & a.name == b.name);

    /*
    struct something x;
    int b = argv[1];
    x.a = argv[2];
    x.b= argv[3];
    x.name = 'b';
    __trident_choice("L9", "i32", (void*[]){&x, &b}, (char*[]){"x", "b"}, 2, (void**[]){&x}, (char*[]){"x"}, 1);
    return TRIDENT_OUTPUT("x", "i32", x.a);
     */

}
