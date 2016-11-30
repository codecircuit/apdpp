#!/bin/bash

# on victoria
CUDA=/opt/cuda-6.5/lib64/stubs
JSONCPP_LIB=/home/cklein/lib
BITOP_LIB=/home/cklein/lib

g++ -o stencil-cpall stencil-cpall.o -L$CUDA -lcuda -L$JSONCPP_LIB -L$BITOP_LIB -ljsoncpp -lbitop
