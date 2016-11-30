#!/bin/bash

# it is mandatory that you are in the script's directory, when you
# execute it.

LIBCLC=${HOME}/src/libclc
DCP=bsp_transform

SCRIPTPATH=$PWD
PROJECTPATH=$PWD/../
PASSBUILDPATH=../build/
PASSPATH=${PASSBUILDPATH}lib/${DCP}.so
TESTPATH=${PROJECTPATH}test_orig/
TESTPATH_BC=${TESTPATH}/llvm/test_device.bc
OUTPATH_BC=${PROJECTPATH}test_trans/llvm/test_device.bc
OUTPATH_LL=${PROJECTPATH}test_trans/llvm/test_device.ll
OUTPATH_PTX=${PROJECTPATH}test_trans/ptx/test_device
OUTPATH_LINKED=${PROJECTPATH}test_trans/llvm/test_device.linked.bc
STRUCTNAME=${DCP}
DB="../bsp_analysis/dbs/kernel_info.ddb"
TARGET=nvptx64--nvidiacl

# making the pass
cd ${PASSBUILDPATH}
make ${DCP}

# making the test case
cd ${TESTPATH}
make llvm/test_device.bc

# going back
cd ${SCRIPTPATH}

# Running the pass
opt -debug -load ${PASSPATH} -${STRUCTNAME} -mekong_db ${DB} < ${TESTPATH_BC} > /dev/null -o ${OUTPATH_BC} 

# creating .ll file
clang++ -c -o ${OUTPATH_LL} -S -emit-llvm ${OUTPATH_BC} -target ${TARGET} -x cl

# linking to nvidiacl
llvm-link ${OUTPATH_BC} -o ${OUTPATH_LINKED} ${LIBCLC}/built_libs/${TARGET}.bc

# creating .ptx file
llc -mcpu=sm_30 -o ${OUTPATH_PTX}_sm_30.ptx ${OUTPATH_LINKED}

# creating .ptx file
clang -target ${TARGET} ${OUTPATH_LINKED} -S -o ${OUTPATH_PTX}_sm_20.ptx
