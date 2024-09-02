# set cmake minimum version
cmake_minimum_required(VERSION 3.20.2)

# set project name
project(flash-attn-101 VERSION 0.0.1 LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=177")

# find cuda toolkit
find_package(CUDAToolkit REQUIRED)

# set include directory
find_path(PROJECT_INCLUDE_DIR cuda_utils.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
set(FLASH_DIR ${CMAKE_SOURCE_DIR}/csrc/flash_attn)
set(FLASH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/csrc/flash_attn/src)
set(CUTE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/csrc/cutlass/include)

# add library
add_library(
    flash-attn-101
    SHARED
    naive_attention.cu
    flash_attn_1.cu
    flash_attn_2.cu
    flash_attn/flash_api.cu
    flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu
    flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu
    cuda_utils.cu
    profile_utils.cu
)
# library include directory
target_include_directories(flash-attn-101 PUBLIC ${PROJECT_INCLUDE_DIR} ${FLASH_DIR} ${FLASH_INCLUDE_DIR} ${CUTE_INCLUDE_DIR})
# set gpu architecture
set_target_properties(flash-attn-101 PROPERTIES CUDA_ARCHITECTURES 80)

# add executable
add_executable(
    profile-attention
    profile.cu
)
# link libraries
target_link_libraries(profile-attention flash-attn-101)
# set gpu architecture
set_target_properties(profile-attention PROPERTIES CUDA_ARCHITECTURES 80)