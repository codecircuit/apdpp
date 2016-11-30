project(bitop)

include_directories(inc)
add_executable(bitop_test src/bitop.cc)
add_library(bitop SHARED src/bitop.cc)

set_target_properties(bitop_test PROPERTIES COMPILE_FLAGS "-std=c++11 -DBITOP")
set_target_properties(bitop PROPERTIES COMPILE_FLAGS "-std=c++11")
