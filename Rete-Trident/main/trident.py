import subprocess

TRIDENT_RUNTIME = "Trident_runtime/"
KLEE_DIR = "'/klee/include'"

EXAMPLE = "test_functions/codeflaws_1.c"
build_ret = subprocess.call("KLEE_INCLUDE_PATH={} make -C ../{}".format(KLEE_DIR, TRIDENT_RUNTIME), shell=True, stdout=subprocess.PIPE)
build_ret2 = subprocess.call("clang -c ../{} -emit-llvm -I .. -o ../test_functions/function.bc".format(EXAMPLE), shell=True, stdout=subprocess.PIPE)
subprocess.call("llvm-link ../test_functions/function.bc ../{}.trident_runtime.o.bc -o ../test_functions/out.bc".format(TRIDENT_RUNTIME), shell=True, stdout=subprocess.PIPE)
subprocess.call("klee --write-kqueries --write-cvcs --external-calls=all ../test_functions/out.bc", shell=True, stdout=subprocess.PIPE)

# Do stuff

subprocess.call("rm -rf ../test_functions/*.bc ../test_functions/klee-* ../test_functions/out.bc", shell=True, stdout=subprocess.PIPE)

