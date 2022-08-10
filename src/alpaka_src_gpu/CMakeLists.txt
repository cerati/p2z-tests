# Required CMake version.

cmake_minimum_required(VERSION 3.15)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME alpaka_gpu_src)

project(${_TARGET_NAME})

#-------------------------------------------------------------------------------
# Find alpaka.

find_package(alpaka REQUIRED)

#-------------------------------------------------------------------------------
# Add executable.

set(CMAKE_CUDA_ARCHITECTURES 70)
#add_compile_definitions(num_streams=1)
add_compile_definitions(EXCLUDE_H2D_TRANSFER=1)
add_compile_definitions(EXCLUDE_D2H_TRANSFER=1)
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
#add_compile_options(-std=c++17)
#add_compile_options(-maxrregcount=128)
add_compile_options(--use_fast_math)

#add_compile_options(-ffast-math)
#add_compile_options(-march=native)
#add_compile_options(-O3)
#add_compile_options(-fopenmp-simd)

set(INCLUDE_DIR
    "~/fermi/p2r-tests/alpaka/include/" 
)
#
include_directories(${INCLUDE_DIR})
alpaka_add_executable(
    ${_TARGET_NAME}
    src/propagate-toz-test_alpaka_cpu_gpu.cpp)

#target_link_directories(${_TARGET_NAME} PUBLIC ~/fermi/p2r-tests/alpaka/install/lib64/)
#target_compile_options(${_TARGET_NAME} PRIVATE -maxrregcount=64)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka
)