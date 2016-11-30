#!/bin/bash

# it is mandatory that you are in the directory of the script
# when you execute it

HCP=host_transform

SCRIPTPATH=$PWD
PROJECTPATH=$PWD/../
PASSBUILDPATH=../build/
PASSPATH=${PASSBUILDPATH}lib/${HCP}.so
TESTPATH=${PROJECTPATH}test_orig/
TESTPATH_BC=${TESTPATH}/llvm/test_host.bc
OUTPATH_BC=${PROJECTPATH}test_trans/llvm/test_host.bc
OUTPATH_LL=${PROJECTPATH}test_trans/llvm/test_host.ll
OUTPATH=${PROJECTPATH}test_trans/bin/test_host
OUTPATH_O=${PROJECTPATH}test_trans/src/test_host.o
STRUCTNAME=${HCP}

# making the pass
cd ../build
make host_transform

# making the test case
cd ../test_orig
make -j4

# going back
cd ../sh

# Running the pass
# -debug-only=LAUNCH_CHANGER
opt -S -debug -load ../build/lib/ins_rt_decls.so -load \
                    ../build/lib/host_transform.so -host_transform \
                    ../test_orig/llvm/test_host.ll -o \
                    ../test_trans/llvm/test_host.ll

# creating .o file
clang++ -c -std=c++11 -o ../test_trans/src/test_host.o ../test_trans/llvm/test_host.ll

# creating binary
clang++ -std=c++11 -o ../test_trans/bin/test_host \
                      ../test_trans/src/test_host.o \
                      ../runtime/build/libmekong-rt.a \
                      -L${ISL_LIB} -lisl \
                      -L${CUDA_LIB} -lcuda \
                      -L../jsoncpp/build/ -ljsoncpp -pthread
