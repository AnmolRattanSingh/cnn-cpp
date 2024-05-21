cmake_minimum_required(VERSION 3.22)

project(HuffmanEncoding
  VERSION 0.1.0
  DESCRIPTION "Convolutional Neural Network implemented in C++"
  LANGUAGES CXX
)

# Set the C++ standard to C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

include_directories(${EIGEN_INCLUDE_DIR})

add_library(convolution convolution.cpp convolution.hpp)
add_library(correlate correlate.cpp correlate.hpp)
add_executable(main main.cpp)

target_link_libraries(main convolution correlate Eigen3::Eigen hdf5::hdf5 hdf5::hdf5_cpp)
