// declare all attention launcher template functions
#pragma once
#include <cuda_runtime.h>

template <typename T>
void launch_naive_attention(
    const T *Q,
    const T *K,
    const T *V,
    T *O,
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
);

template <typename T>
void launch_flash_attention_01(
    const T *Q,
    const T *K,
    const T *V,
    T *O,
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
);

template <typename T>
void launch_flash_attention_02(
    const T *Q,
    const T *K,
    const T *V,
    T *O,
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
);

template <typename T>
void mha_fwd(
    const T *Q,
    const T *K,
    const T *V,
    T *O,
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
);