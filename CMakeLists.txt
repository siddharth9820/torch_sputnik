cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(torch_sputnik)

find_package(Torch REQUIRED)
find_package(ATen REQUIRED)
find_package(CUDA 11.0 REQUIRED)
#/ccs/home/ssingh37/sputnik

#add_library( sputnik SHARED IMPORTED )
link_directories(${SPUTNIK_DIR}/build/sputnik)
include_directories(${SPUTNIK_DIR})

add_library(sputnik_ops SHARED op.cpp)
target_link_libraries(sputnik_ops "${TORCH_LIBRARIES}")
target_link_libraries(sputnik_ops sputnik)
target_link_libraries(sputnik_ops ${CUDA_LIBRARIES})
set_property(TARGET sputnik_ops PROPERTY CXX_STANDARD 14)
