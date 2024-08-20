# set cmake minimum version
cmake_minimum_required(VERSION 3.20.2)

# set project name
project(flash-attn-101 VERSION 0.0.1 LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find cuda toolkit
find_package(CUDAToolkit REQUIRED)

# set include directory
find_path(PROJECT_INCLUDE_DIR cuda_utils.hpp HINTS ${CMAKE_SOURCE_DIR}/include)

# add library
add_library(
    flash-attn-101
    SHARED
    naive_attention.cu
    flash_attn_1.cu
    flash_attn_2.cu
    cuda_utils.cu
    profile_utils.cu
)
# library include directory
target_include_directories(flash-attn-101 PUBLIC ${PROJECT_INCLUDE_DIR})
# set gpu architecture
set_target_properties(flash-attn-101 PROPERTIES CUDA_ARCHITECTURES 70)

# add executable
add_executable(
    profile-attention
    profile.cu
)
# link libraries
target_link_libraries(profile-attention flash-attn-101)
# set gpu architecture
set_target_properties(profile-attention PROPERTIES CUDA_ARCHITECTURES 70)