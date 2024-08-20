#include <iostream>

#include "profile_utils.cuh"
#include "cuda_utils.hpp"

void print_device_info()
{   
    // get device
    int device_id = 0;
    cudaGetDevice(&device_id);
    
    // get device properties
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    // print device name
    std::cout << "Device name: " << device_prop.name << std::endl;

    // print global memory size in GB
    float const global_memory_size{static_cast<float>(device_prop.totalGlobalMem / (1 << 30))}; // 1 << 30 bytes = 1 GiB ~ 1 GB 
    std::cout << "Global memory size: " << global_memory_size << " GB" << std::endl;

    // print peak memory bandwidth
    float const peak_bandwidth{static_cast<float>((2.0f * device_prop.memoryClockRate * device_prop.memoryBusWidth / 8) / 1.0e6)}; // 1.0e6 from kHz to Ghz
    std::cout << "Peak memory bandwidth: " << peak_bandwidth << " GB/s" << std::endl;

    std::cout << std::endl;
}

float measure_performance(
    std::function<void(cudaStream_t)> launcher, 
    cudaStream_t stream, 
    const int num_warmups, 
    const int num_repeats)
{
    // Event initialization
    float milliseconds = 0.0f;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // warmup
    for (unsigned int i = 0; i < num_warmups; i++)
    {
        launcher(stream);
    }

    // synchronize
    cudaStreamSynchronize(stream);

    // benchmark
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i = 0; i < num_repeats; i++)
    {
        launcher(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));

    // synchronize
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();

    // get result and destroy events
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return milliseconds / static_cast<float>(num_repeats);
}

