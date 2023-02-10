CC=gcc
CFLAGS=-I $(KLEE_INCLUDE_PATH)

DEP = trident_runtime.h
SRC = trident_runtime.c
MERGE = klee_merge_placeholder.c
MERGE_DEP = klee_merge_placeholder.h
OBJ_KLEE = $(SRC:%.c=%.o)
OBJ_MERGE = $(MERGE:%.c=%.o)
all: trident_runtime.o libtrident_runtime.a klee_merge_placeholder.o libklee_merge.a

%.o: %.c $(DEP)
	wllvm -g -O0 -c -o $@ $< $(CFLAGS)

%.o: %.c $(MERGE_DEP)
	wllvm -g -O0 -c -o $@ $< $(CFLAGS)

libtrident_runtime.a: $(OBJ_KLEE)
	llvm-ar rcs $@ $(OBJ_KLEE)

libklee_merge.a: $(OBJ_MERGE)
	llvm-ar rcs $@ $(OBJ_MERGE)

clean:
	rm -f *.o
	rm -f *.a