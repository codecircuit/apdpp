##################################################################
#
# DESCRIPTION - Makefile for N Body Code
#
# AUTHOR      - Christoph Klein
#
# LAST CHANGE - 2016-06-30
#
##################################################################

# If you change the compile system change the
# library paths here
CPP=clang++

LIB=-L$(CUDA_LIB) -lcuda 
CFLAGS=-std=c++11
INC=-I$(CUDA_INC)
OPENCL_LIB=$(CLC_LIB)
OPENCL_INC=$(CLC_INC)

OBJ=src/test_host.o

NAMEH=test_host
NAMED=test_device

TARGET=nvptx64--nvidiacl

RED=\033[0;31m
NC=\033[0m
GREEN=\033[0;32m

.PHONY:
all: bin/$(NAMEH) llvm/$(NAMEH).ll llvm/$(NAMED).bc ptx/$(NAMED)_sm_20.ptx ptx/$(NAMED)_sm_30.ptx llvm/$(NAMEH).bc llvm/$(NAMED).linked.bc llvm/$(NAMED).ll

.PHONY:
show_includes:
	@printf "$(INC)\n\n"

bin/$(NAMEH): $(OBJ)
	@printf "$(RED)LINKING $(NC)[$(GREEN)$(NAMEH)$(NC)]\n"
	@$(CPP) $(CFLAGS) $(INC) -o bin/$(NAMEH) src/test_host.o $(LIB)

llvm/$(NAMEH).bc: src/$(NAMEH).cc
	@printf "$(RED)CREATING LLVM-IR (.bc) $(NC)[$(GREEN)$(NAMEH)$(NC)]\n"
	@$(CPP) $(CFLAGS) $(INC) -c -emit-llvm src/*.cc -o llvm/$(NAMEH).bc

llvm/$(NAMEH).ll: src/$(NAMEH).cc
	@printf "$(RED)CREATING LLVM-IR (.ll) $(NC)[$(GREEN)$(NAMEH)$(NC)]\n"
	@$(CPP) $(CFLAGS) $(INC) -c -S -emit-llvm src/*.cc -o llvm/$(NAMEH).ll 

llvm/$(NAMED).bc: src/$(NAMED).cl
	@printf "$(RED)CREATING LLVM-IR (.bc) $(NC)[$(GREEN)$(NAMED)$(NC)]\n"
	@clang -Dcl_clang_storage_class_specifiers -include $(OPENCL_INC)/clc/clc.h \
	      -isystem $(OPENCL_INC) -target $(TARGET)\
	      -x cl -c src/$(NAMED).cl -emit-llvm -o llvm/$(NAMED).bc

llvm/$(NAMED).linked.bc: llvm/$(NAMED).bc
	@printf "$(RED)LINKING $(NC)[$(GREEN)$(NAMED)$(NC)]\n"
	@llvm-link llvm/$(NAMED).bc -o llvm/$(NAMED).linked.bc $(OPENCL_LIB)/built_libs/$(TARGET).bc 

llvm/$(NAMED).ll: src/$(NAMED).cl
	@printf "$(RED)CREATING LLVM-IR (.ll) $(NC)[$(GREEN)$(NAMED)$(NC)]\n"
	@clang -Dcl_clang_storage_class_specifiers -include $(OPENCL_INC)/clc/clc.h \
	      -isystem $(OPENCL_INC) -target $(TARGET)\
	      -x cl -c src/$(NAMED).cl -S -emit-llvm -o llvm/$(NAMED).ll

ptx/$(NAMED)_sm_20.ptx: llvm/$(NAMED).linked.bc llvm/$(NAMED).ll
	@mkdir -p ptx
	@printf "$(RED)LINKING PTX $(NC)[$(GREEN)$(NAMED)$(NC)]\n"
	@clang -target $(TARGET) llvm/$(NAMED).linked.bc -S -o ptx/$(NAMED)_sm_20.ptx

ptx/$(NAMED)_sm_30.ptx: llvm/$(NAMED).linked.bc llvm/$(NAMED).ll
	@mkdir -p ptx
	@printf "$(RED)LINKING PTX $(NC)[$(GREEN)$(NAMED)$(NC)]\n"
	@llc -mcpu=sm_30 llvm/$(NAMED).linked.bc -o ptx/$(NAMED)_sm_30.ptx

src/test_host.o: src/test_host.cc
	@printf "$(RED)COMPILING $(NC)[$(GREEN)$(NAMEH)$(NC)]\n"
	@clang++ $(INC) $(CFLAGS) -c -o src/test_host.o src/test_host.cc

.PHONY: clean
clean :
	@printf "$(RED)REMOVING OBEJCT AND BINARY FILES$(NC)\n"
	rm -f src/*.o bin/$(NAMEH)
	@printf "$(RED)REMOVING LLVM-IR$(NC)\n"
	rm -f llvm/*.ll llvm/*.bc
	@printf "$(RED)REMOVING PTX$(NC)\n"
	rm -f ptx/*.ptx
	
.PHONY: rebuild
rebuild:
	make clean
	make
