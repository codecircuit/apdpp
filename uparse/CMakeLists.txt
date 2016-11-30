cmake_minimum_required(VERSION 3.1.3)
project(uparse)

include_directories(inc)
add_executable(uparse_test src/uparse.cc)
add_library(uparse SHARED src/uparse.cc)

set_target_properties(uparse_test PROPERTIES COMPILE_FLAGS "-std=c++11 -DTEST")
set_target_properties(uparse PROPERTIES COMPILE_FLAGS "-std=c++11")
