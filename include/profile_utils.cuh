// implement all profile utils
#pragma once

#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <functional>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include "cuda_utils.hpp"
#include "cuda_attn.hpp"

// print helper functions
void print_device_info();

// random matrix generator
template <typename T>
void initialize_random_matrix(T* A, unsigned int size, unsigned int seed)
{
    // create a generator
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    auto const generator = [&engine, &distribution]() {
        return distribution(engine);
    };

    // initialize matrix
    for (unsigned int i = 0; i < size; i++) {
        A[i] = static_cast<T>(generator());
    }
}

// all-close check
template <typename T>
bool all_close(
    const T* A, const T* A_ref,
    unsigned int size,
    T abs_tol, double rel_tol
)
{
    // cast all computation into double for accurate check
    bool status = true;
    for (unsigned int i = 0; i < size; i++)
    {
        const double A_val = static_cast<double>(A[i]);
        const double A_ref_val = static_cast<double>(A_ref[i]);
        const double diff = A_val - A_ref_val;
        if (diff > std::max(static_cast<double>(abs_tol), rel_tol * static_cast<double>(std::abs(A_ref_val))))
        {
            std:: cout << "A[" << i << "] = " << A_val << ", A_ref[" << i << "] = " << A_ref_val << std::endl;
            std:: cout << "diff = " << diff << " > " << std::max(static_cast<double>(abs_tol), rel_tol * static_cast<double>(std::abs(A_ref_val))) << std::endl;
            status = false;
            return status;
        }
    }

    return status;
}

// benchmark function
float measure_performance(
    std::function<void(cudaStream_t)> launcher,
    cudaStream_t stream,
    const int num_warmups,
    const int num_repeats
);

// profile function
template <typename T>
std::pair<float, float> profile_attention(
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    std::function<void(
        const T *,
        const T *,
        const T *,
        T *,
        unsigned int, unsigned int , unsigned int, unsigned int,
        cudaStream_t
    )> attention_launcher,
    unsigned int num_warmups, unsigned int num_repeats,
    T abs_tol, double rel_tol,
    unsigned int seed = 0
)
{
    // create stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // allocate memory on host
    thrust::host_vector<T> h_Q(batch_size * num_heads * seq_len * head_dim);
    thrust::host_vector<T> h_K(batch_size * num_heads * seq_len * head_dim);
    thrust::host_vector<T> h_V(batch_size * num_heads * seq_len * head_dim);
    thrust::host_vector<T> h_O(batch_size * num_heads * seq_len * head_dim);
    thrust::host_vector<T> h_O_ref(batch_size * num_heads * seq_len * head_dim);
    T *h_Q_ptr = h_Q.data();
    T *h_K_ptr = h_K.data();
    T *h_V_ptr = h_V.data();
    T *h_O_ptr = h_O.data();
    T *h_O_ref_ptr = h_O_ref.data();
    CHECK_LAST_CUDA_ERROR();

    // initialize matrix
    unsigned int problem_size = batch_size * num_heads * seq_len * head_dim;
    initialize_random_matrix<T>(h_Q_ptr, problem_size, seed);
    initialize_random_matrix<T>(h_K_ptr, problem_size, seed + 1);
    initialize_random_matrix<T>(h_V_ptr, problem_size, seed + 2);
    thrust::fill(h_O.begin(), h_O.end(), static_cast<T>(0.0f));
    thrust::fill(h_O_ref.begin(), h_O_ref.end(), static_cast<T>(0.0f));

    // allocate memory on device and copy to device
    thrust::device_vector<T> d_Q(h_Q);
    thrust::device_vector<T> d_K(h_K);
    thrust::device_vector<T> d_V(h_V);
    thrust::device_vector<T> d_O(h_O);
    thrust::device_vector<T> d_O_ref(h_O_ref);
    T *d_Q_ptr = d_Q.data().get();
    T *d_K_ptr = d_K.data().get();
    T *d_V_ptr = d_V.data().get();
    T *d_O_ptr = d_O.data().get();
    T *d_O_ref_ptr = d_O_ref.data().get();
    CHECK_LAST_CUDA_ERROR();

    // all-close sanity check
    launch_naive_attention<T>(d_Q_ptr, d_K_ptr, d_V_ptr, d_O_ref_ptr, batch_size, num_heads, seq_len, head_dim, stream);
    cudaStreamSynchronize(stream);
    // copy d_O_ref to h_O_ref
    thrust::copy(d_O_ref.begin(), d_O_ref.end(), h_O_ref.begin());

    attention_launcher(d_Q_ptr, d_K_ptr, d_V_ptr, d_O_ptr, batch_size, num_heads, seq_len, head_dim, stream);
    cudaStreamSynchronize(stream);
    // copy d_O to h_O
    thrust::copy(d_O.begin(), d_O.end(), h_O.begin());

    assert(all_close(h_O_ptr, h_O_ref_ptr, problem_size, abs_tol, rel_tol));
    std::cout << "all-close check passed" << std::endl;

    // performance benchmark
    float naive_latency = measure_performance(
        [&](cudaStream_t stream) {
            launch_naive_attention<T>(d_Q_ptr, d_K_ptr, d_V_ptr, d_O_ref_ptr, batch_size, num_heads, seq_len, head_dim, stream);
            return;
        },
        stream, num_warmups, num_repeats
    );
    float custom_latency = measure_performance(
        [&](cudaStream_t stream) {
            attention_launcher(d_Q_ptr, d_K_ptr, d_V_ptr, d_O_ptr, batch_size, num_heads, seq_len, head_dim, stream);
            return;
        },
        stream, num_warmups, num_repeats
    );

    // print results
    std::cout << "naive attention latency = " << naive_latency << " ms" << std::endl;
    std::cout << "latency = " << custom_latency << " ms" << std::endl;
    std::cout << "speedup = " << naive_latency / custom_latency * 100.0f << "%" << std::endl;

    return {naive_latency, custom_latency};
}