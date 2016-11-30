#!/bin/bash

PROJECTPATH=${PWD}/..
NEWNAME=kernel.ptx
TARGETDIR=~/tmp/
HOST=$1
DCA=bsp_analysis
OBJECTPATH=${PROJECTPATH}/test_trans/src/test_host.o
MEKONG_LIB=../runtime/build/libmekong-rt.a

if [[ "$HOST" == "creek05" ]]; then
	PTXPATH=${PROJECTPATH}/test_trans/ptx/test_device_sm_20.ptx
	CUDA_LIB=/opt/cuda-7.5/lib64/stubs
	echo "host is a creek node"
fi
if [[ "$HOST" == "victoria" ]]; then
	PTXPATH=${PROJECTPATH}/test_trans/ptx/test_device_sm_30.ptx
	CUDA_LIB=/opt/cuda-6.5/lib64/stubs
	echo "host is victoria"
fi

JSONCPP_LIB=/home/cklein/lib
#BITOP_LIB=/home/cklein/lib
#PARSER_LIB=/home/cklein/lib
ISL_LIB=/home/cklein/lib
cd ../runtime/build
make mekong-rt -j4
cd ../../sh
scp ${PTXPATH} ${HOST}:${TARGETDIR}${NEWNAME}
scp ${MEKONG_LIB} ${HOST}:${TARGETDIR}
scp ${OBJECTPATH} ${HOST}:${TARGETDIR}
ssh ${HOST} ''g++ -o ${TARGETDIR}test_host ${TARGETDIR}test_host.o ${TARGETDIR}libmekong-rt.a -L${CUDA_LIB} -lcuda -L${ISL_LIB} -lisl -L${JSONCPP_LIB} -lsofire -ljsoncpp -pthread''
