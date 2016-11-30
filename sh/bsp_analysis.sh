#!/bin/bash

# it is mandatory that you are in the script's directory, when you
# execute it.

out=$(opt -help | grep polly)
if [ -z "$out" ]; then
	pollyLoaded=false
else
	pollyLoaded=true
fi

LIBCLC=${HOME}/src/libclc

DCA=bsp_analysis
DB="../${DCA}/dbs/kernel_info.ddb"
echo "Database output: ${DB}"
SCRIPTPATH=$PWD
PROJECTPATH=$PWD/../
PASSBUILDPATH=../build/
PASSPATH=${PASSBUILDPATH}lib/${DCA}.so
LWPASSPATH=${PASSBUILDPATH}lib/loop_wrapping_pass.so
TESTPATH=${PROJECTPATH}test_orig/
TESTPATH_BC=${TESTPATH}/llvm/test_device.bc
TESTPATH_LL=${TESTPATH}/llvm/test_device.ll
TESTPATH_WRAPPED=${TESTPATH}/llvm/test_device.wrapped.ll
STRUCTNAME=${DCA}

# making the pass
cd ${PASSBUILDPATH}
make ${DCA}

# making the test case
cd ${TESTPATH}
make llvm/test_device.ll

# going back
cd ${SCRIPTPATH}

# Running the pass
# -debug-only=PROMOTER
POLLY_LIB="/home/cklein/build/llvm_git/lib/LLVMPolly.so"
TESTPATH_CANONIC_LL="${TESTPATH_LL%.*}.canonic.ll"
if [ "$pollyLoaded" = true ]; then
	opt -S -load $LWPASSPATH -lwpass $TESTPATH_LL > $TESTPATH_WRAPPED 
	opt -S -polly-canonicalize $TESTPATH_WRAPPED > ${TESTPATH_CANONIC_LL}
	opt -load ${PASSPATH} -${STRUCTNAME} -mekong_db ${DB} ${TESTPATH_CANONIC_LL} > /dev/null
else
	opt -S -load $LWPASSPATH -lwpass $TESTPATH_LL > $TESTPATH_WRAPPED 
	opt -load ${POLLY_LIB} -S -polly-canonicalize $TESTPATH_WRAPPED > $TESTPATH_CANONIC_LL
	opt -load ${POLLY_LIB} -load ${PASSPATH} -${STRUCTNAME} -mekong_db ${DB} ${TESTPATH_CANONIC_LL} > /dev/null
fi
#opt -debug -load ${POLLY_LIB} -load ${PASSPATH} -${STRUCTNAME} -mekong_db ${DB} ${TESTPATH_BC} > /dev/null
