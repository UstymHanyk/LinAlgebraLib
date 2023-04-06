cmake_minimum_required(VERSION 3.20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
# Set the compiler to GCC or Clang
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

project(LinAlgLib)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
set (CMAKE_CXX_FLAGS "-mavx -fopenmp")

add_executable(LinAlgLib main.cpp Vector.h Matrix.h)

# Link the OpenMP library to your target executable
target_link_libraries(LinAlgLib OpenMP::OpenMP_CXX)
