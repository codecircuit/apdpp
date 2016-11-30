cmake_minimum_required(VERSION 3.1.3)
project(Mekong)

# SHORTCUTS
set(DCA bsp_analysis)
set(DCP bsp_transform)
set(HCP host_transform)

# FIND CUDA
#find_package(CUDA REQUIRED)
#message(STATUS "found cuda =  ${CUDA_FOUND}")
#message(STATUS "cuda include dirs = ${CUDA_INCLUDE_DIRS}")

# LOAD LLVM MODULES
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in : ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
add_definitions(${LLVM_DEFINITIONS})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(LLVM_ENABLE_PLUGINS ON)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(LLVM_PLUGIN_EXT ".so")

# INCLUDE POLLY
# TODO find a consistent solution to find polly
find_library(POLLY_LIB NAMES LLVMPolly.so)
find_path(POLLY_INC NAMES "polly/ScopInfo.h")
if (EXISTS ${POLLY_LIB})
	message(STATUS "Found polly library in: ${POLLY_LIB}")
else (EXISTS ${POLLY_LIB})
	message(FATAL_ERROR "Could not found polly library LLVMPolly.so! Set your CMAKE_LIBRARY_PATH variable appropriately!")
endif(EXISTS ${POLLY_LIB})
if (EXISTS ${POLLY_INC})
	message(STATUS "Found polly include in: ${POLLY_INC}")
else (EXISTS ${POLLY_INC})
	message(FATAL_ERROR "Could not found polly include dir! Set your CMAKE_INCLUDE_PATH variable appropriately!")
endif(EXISTS ${POLLY_INC})

# INCLUDE POLLY's INTERNAL ISL LIBRARY
find_library(ISL_LIB NAMES libPollyISL.a)
set(ISL_INC "${POLLY_INC}/../lib/External/isl/include")
if (EXISTS ${ISL_LIB})
	message(STATUS "found Polly's ISL lib in: ${ISL_LIB}")
else (EXISTS ${ISL_LIB})
	message(FATAL_ERROR "could not found polly's internal ISL lib: libPollyISL.a! Set your CMAKE_LIBRARY_PATH appropriately!")
endif(EXISTS ${ISL_LIB})
if (EXISTS ${ISL_INC})
	message(STATUS "found Polly's ISL include dir in: ${ISL_INC}")
else (EXISTS ${ISL_INC})
	message(FATAL_ERROR "could not found polly's internal ISL include dir! Adjust the path in the CMakeLists.txt!")
endif(EXISTS ${ISL_INC})

# SET LINK DIRECTORIES
link_directories(${LLVM_LIBRARY_DIRS})

# INSTALL EVERYTHING LOCAL
set(CMAKE_INSTALL_PREFIX ./)

# INCLUDE NEEDED DIRECTORIES
include_directories(${LLVM_INCLUDE_DIRS}
                    ./${DCA}/inc
                    ${CUDA_INCLUDE_DIRS}
                    ${POLLY_INC}
                    ${ISL_INC}
                    dashdb/inc)

# DEVICE CODE ANALYSIS
add_llvm_loadable_module(${DCA} ./${DCA}/src/${DCA}.cc dashdb/src/dashdb.cc)
add_llvm_loadable_module(loop_wrapping_pass ./${DCA}/src/lwpass.cc)
set_target_properties(${DCA} PROPERTIES COMPILE_FLAGS "-std=c++11 -fno-rtti")
set_target_properties(loop_wrapping_pass PROPERTIES COMPILE_FLAGS "-std=c++11 -fno-rtti")

# DEVICE CODE PASS
set(DCPSRC ${DCP}/src/)
add_llvm_loadable_module(${DCP} ${DCPSRC}${DCP}.cc ${DCPSRC}promoter.cc ${DCPSRC}propagator.cc ${DCPSRC}offsetter.cc ${DCPSRC}pattern.cc dashdb/src/dashdb.cc)
set_target_properties(${DCP} PROPERTIES COMPILE_FLAGS "-std=c++11 -fno-rtti")

# HOST CODE PASS
set(HCPSRC ${HCP}/src/)
add_llvm_loadable_module(${HCP} ${HCPSRC}/${HCP}.cc)
add_llvm_loadable_module(ins_rt_decls ${HCPSRC}/ins_rt_decls.cc)
set_target_properties(${HCP} PROPERTIES COMPILE_FLAGS "-std=c++11 -fno-rtti")
set_target_properties(ins_rt_decls PROPERTIES COMPILE_FLAGS "-std=c++11 -fno-rtti")

add_subdirectory(bitop)

target_link_libraries(${DCA} ${ISL_LIB})

# INSTALL FILES
install(PROGRAMS ./driver/alleycatcc.py DESTINATION ./)
install(FILES ./alleycat/src/alleycat.h ./bitop/inc/bitop.h DESTINATION ./inc)
install(FILES ./alleycat/src/kernel_info.h DESTINATION ./inc)
