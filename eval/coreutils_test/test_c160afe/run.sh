#!/usr/bin/env bash

PROGRAM=test

rm -rf klee-t*
mkdir -p -m ug-s,u=rwx,g=rwx,o=rx a/c
mkdir -p -m ug-s,u=rwx,g=,o=  d/c


save=$(pwd)

cd /home/Trident/Trident/runtime || return
KLEE_INCLUDE_PATH=/klee/include make

cd $save || return
cd .. || return
cd coreutils_c160afe || return
#autoreconf -f -i
./configure CFLAGS="-g -O0" CC=wllvm
make clean
if [ $1 == "trident" ]
then
  patch  src/cp.c "${save}"/trident.transform
else
  patch  src/cp.c "${save}"/semfix.transform
fi

make CC=wllvm

echo "Start linking"
wllvm -include /home/Trident/Trident/runtime/trident_runtime.h -g -O0 -o src/cp src/cp.o src/copy.o src/cp-hash.o src/extent-scan.o src/force-link.o src/selinux.o src/libver.a lib/libcoreutils.a lib/libcoreutils.a -L/klee/build/lib -L/home/Trident/Trident/runtime/ -ltrident_runtime -lkleeRuntest -lklee_merge
cd src || exit

extract-bc cp

CP=cp.bc

echo "Compiled!"

if [ $1 == "klee-merge" ]
then
    klee --posix-runtime --libc=uclibc --write-smt2s -use-merge --max-solver-time 100 --max-time 1200 --max-forks=50 --output-dir="${save}"/klee-t1 "${CP}" -a "${save}"/a "${save}"/d
    klee --posix-runtime --libc=uclibc --write-smt2s -use-merge --max-solver-time 100 --max-time 1200 --max-forks=50 --output-dir="${save}"/klee-t2 "${CP}" -r "${save}"/a "${save}"/d
    klee --posix-runtime --libc=uclibc --write-smt2s -use-merge --max-solver-time 100 --max-time 1200 --max-forks=50 --output-dir="${save}"/klee-t3 "${CP}" --attributes-only -a "${save}"/a "${save}"/d
else
    klee --posix-runtime --libc=uclibc --write-smt2s --max-solver-time 100 --max-time 1200 --max-forks=50 --output-dir="${save}"/klee-t1 "${CP}" -a "${save}"/a "${save}"/d
    klee --posix-runtime --libc=uclibc --write-smt2s --max-solver-time 100 --max-time 1200 --max-forks=50 --output-dir="${save}"/klee-t2 "${CP}" -r "${save}"/a "${save}"/d
    klee --posix-runtime --libc=uclibc --write-smt2s --max-solver-time 100 --max-time 1200 --max-forks=50 --output-dir="${save}"/klee-t3 "${CP}" --attributes-only -a "${save}"/a "${save}"/d
fi

patch -R cp.c "${save}"/trident.transform


cd $save || exit


klee --posix-runtime --libc=uclibc --write-smt2s --output-dir=klee-t1 "${PROGRAM}.bc" -E -w -f /dev/null  > /dev/null 2>&1
klee --posix-runtime --libc=uclibc --write-smt2s --output-dir=klee-t2 "${PROGRAM}.bc"


timeout -k 1200s 1200s python3.6 /home/Trident/Trident/main/synthesis.py \
          --tests t1.smt2:klee-t1 t2.smt2:klee-t2 t3.smt2:klee-t3 --priority \
          --components /home/Trident/Trident/components/*.smt2 --theta 3.243 --batch 14