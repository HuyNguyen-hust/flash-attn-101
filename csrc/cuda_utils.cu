#include <iostream>
#include <cuda_runtime.h>

#include "cuda_utils.hpp"

void check_cuda_error(cudaError_t val, const char* const func, const char* const file, const int line)
{
    if (val != cudaSuccess)
    {
        std::cerr << "CUDA runtime error at: " << file << ":" << line << std::endl;
        std::cerr << "CUDA error message: " << cudaGetErrorString(val) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void check_last_cuda_error(const char* const file, const int line)
{
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA runtime error at: " << file << ":" << line << std::endl;
        std::cerr << "CUDA error message: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}