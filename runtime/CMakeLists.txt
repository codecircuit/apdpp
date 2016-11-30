project(MekongRuntime)
cmake_policy(SET CMP0003 NEW)
cmake_minimum_required(VERSION 2.8)

find_library(CUDA_LIB cuda REQUIRED)
find_library(ISL_LIB isl REQUIRED)
find_path(CUDA_INC cuda.h REQUIRED)
find_path(ISL_INC isl/ctx.h REQUIRED)

message(STATUS "Current cmake source directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "Current cmake binary directory: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CUDA library from: ${CUDA_LIB}")
message(STATUS "CUDA include directory: ${CUDA_INC}")
message(STATUS "ISL library from: ${ISL_LIB}")
message(STATUS "ISL include directory: ${ISL_INC}")

# shortcut
set(DCA bsp_analysis)
set(CSD ${CMAKE_SOURCE_DIR})
set(CCBD ${CMAKE_CURRENT_BINARY_DIR})

include_directories(../jsoncpp/inc
                    src
                    ${CUDA_INC}
                    ${CCBD}
                    ../bitop/inc
                    ${ISL_INC}
                    ../uparse/inc
                    ../dashdb/inc
                    ../sofire)

# Generate kernel analysis output header
add_custom_command(
	OUTPUT ${CCBD}/bsp_database.h
	COMMAND ${CMAKE_SOURCE_DIR}/../driver/dashdb2source.sh
            ${CMAKE_SOURCE_DIR}/../${DCA}/dbs/kernel_info.ddb > ${CCBD}/bsp_database.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/../${DCA}/dbs/kernel_info.ddb
)

# Generate user config header
add_custom_command(
	OUTPUT ${CCBD}/user_config.h
	COMMAND ${CMAKE_SOURCE_DIR}/../driver/config2source.sh
            ${CMAKE_SOURCE_DIR}/../CONFIG.txt > ${CCBD}/user_config.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/../CONFIG.txt
)

link_directories(${CUDA_LIB} ${ISL_LIB})

# SET SOURCE FILES
set(MEKONG_RT_SRC
	"src/access_function.cc"
	"src/alias_handle.cc"
	"src/argument_access.cc"
	"src/argument.cc"
	"src/argument_type.cc"
	"src/dependency_resolution.cc"
	"src/log_statistics.cc"
	"src/kernel_info.cc"
	"src/kernel_launch.cc"
	"src/mekong-cuda.cc"
	"src/memory_copy.cc"
	"src/partition.cc"
	"src/partitioning.cc"
	"src/virtual_buffer.cc")

# ADD TEST EXECUTABLES
add_executable(test_partition EXCLUDE_FROM_ALL src/test/test_partition.cc
                                               src/partitioning.cc
                                               src/partition.cc
                                               src/alias_handle.cc
)
add_executable(test_kernellaunch EXCLUDE_FROM_ALL src/test/test_kernellaunch.cc
                                                  src/partitioning.cc
                                                  src/partition.cc
                                                  src/alias_handle.cc
                                                  src/argument.cc
                                                  ../bitop/src/bitop.cc
                                                  ../uparse/src/uparse.cc
                                                  ../dashdb/src/dashdb.cc
                                                  src/argument_type.cc
                                                  src/access_function.cc
                                                  src/argument_access.cc
                                                  src/kernel_info.cc
                                                  src/memory_copy.cc
                                                  src/kernel_launch.cc
                                                  src/mekong-cuda.cc
                                                  ${CCBD}/user_config.h
)

# ADD STATIC RUNTIME LIBRARY
add_library(mekong-rt STATIC "src/mekong-wrapping.cc" ${MEKONG_RT_SRC}
                             ../bitop/src/bitop.cc
                             ../uparse/src/uparse.cc
                             ../dashdb/src/dashdb.cc
                             ${CCBD}/bsp_database.h
                             ${CCBD}/user_config.h)


# STATIC RUNTIME LIBRARY
#   -DSOFIRE for the usage of dominiks memcpy library, which has external linkage
set_target_properties(mekong-rt PROPERTIES
                      COMPILE_FLAGS "-std=c++11 -Wreturn-type -O3")

# TEST CASES

## Partition
set_target_properties(test_partition PROPERTIES
                      COMPILE_FLAGS "-std=c++11 -DMEKONG_TEST -Wreturn-type ")

## Kernel Launch
set_target_properties(test_kernellaunch PROPERTIES
                      COMPILE_FLAGS "-std=c++11 -DMEKONG_TEST -Wreturn-type ")
target_link_libraries(test_kernellaunch ${CUDA_LIB} ${ISL_LIB})
