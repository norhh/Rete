#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef TRIDENT_OUTPUT
#define TRIDENT_OUTPUT(id, typestr, value) value
#endif

void crazy_function(char *p) {
    p[strlen(p) - 1] = '\0';
}
int main(int argc, char *argv[]) {
    char x[100], y[100];
    /*
    klee_make_symbolic(&x, sizeof(x), "value_arg1_a");
    klee_make_symbolic(&y, sizeof(y), "value_arg1_a_DONE");

    crazy_function(&x);
    klee_assume(strcmp(x, y) == 0);
    */
    strcpy(x, "ddd\0");
    __trident_choice("9", (int*[]){&x}, (bool*[]){}, (char*[]){"x"},(char*[]){}, 1 ,0, (int*[]){&x}, (bool*[]){}, (char*[]){"x"},(char*[]){}, 1 ,0);
    return TRIDENT_OUTPUT("x", "i32", strcmp(x, "d"));

}
