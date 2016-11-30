cmake_minimum_required(VERSION 3.1.3)
project(dashdb)

include_directories(inc)
add_executable(dashdb_test src/dashdb.cc)
add_library(dashdb SHARED src/dashdb.cc)

set_target_properties(dashdb_test PROPERTIES COMPILE_FLAGS "-std=c++11 -DTEST")
set_target_properties(dashdb PROPERTIES COMPILE_FLAGS "-std=c++11")
