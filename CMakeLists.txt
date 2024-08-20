# set cmake minimum version
cmake_minimum_required(VERSION 3.20.2)

# set project name
project(flash-attn-101 VERSION 0.0.1 LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add subdirectory
add_subdirectory(csrc)