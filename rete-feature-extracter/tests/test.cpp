//
// Created by Nikhil Parasaram on 30/04/2021.
//

#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[]) {
    int n, i, a[100];
    n = 10+atoi(argv[1]);
    i=1;
    a[10]=5;
    n = atoi(argv[1]) + (n + i + a[10]);
    n +=i+=n;
    for (i=0; i < n; i++) {
        printf("%d\n", i);
    }
    for (int j=n; j < n; j++) {
        printf("%d\n", j);
    }
    while(n--) printf("%d\n", n);
    return 0;
}
