#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>

#include "cuda_attn.hpp"

// kernel

// gemm: aA@B + bC, A (m, k), B(n, k), C(m, n)
// m: seq_len, n: seq_len, k: head_dim
template <typename T>
__global__ void naive_gemm(
    const T* A,
    const T* B,
    T* C,
    T a,
    T b,
    unsigned int M,
    unsigned int N,
    unsigned int K
) 
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int j = 0; j < N; j++) {
        if (j > threadIdx.x) {
            C[idx * N + j] = -INFINITY;
        }
        else {
            T sum = static_cast<T>(0.f);
            for (int k = 0; k < K; k++) {
                sum += A[idx * K + k] * B[(blockDim.x * blockIdx.x + j) * K + k];
            }
            C[idx * N + j] = a * sum + b * C[idx * N + j];
        }
    }
}

// softmax: exp(x - max(x)) / sum(exp(x - max(x)))
template <typename T>
__global__ void naive_softmax(
    T* input,
    T* output,
    unsigned int N
)
{
    // 3-pass softmax
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    T row_max = -INFINITY;
    T sum = static_cast<T>(0.f);

    // max
    for (int j = 0; j < N; j++) {
        row_max = row_max > input[idx * N + j] ? row_max : input[idx * N + j];
    }  

    // sum
    for (int j = 0; j < N; j++) {
        if (j > threadIdx.x) {
            output[idx * N + j] = static_cast<T>(0.f);
        }
        else {
        output[idx * N + j] = __expf(input[idx * N + j] - row_max);
        }
        sum += output[idx * N + j];
        // sum += exp(input[i] - row_max); is not correct because input[i] is also output[i]
    }

    // softmax
    for (int j = 0; j < N; j++) {
        output[idx * N + j] /= sum;
    }
}

// QK[M, M] @ V[M, N]
template <typename T>
__global__ void naive_pv(
    const T *P,
    const T *V,
    T *O,
    unsigned int M, unsigned int N
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int j = 0; j < N; j++) {
        T sum = static_cast<T>(0.f);
        for (int m = 0; m < M; m++) {
            sum += P[idx * M + m] * V[(blockDim.x * blockIdx.x + m) * N + j];
        }
        O[idx * N + j] = sum;
    }
}

// launch
template <typename T>
void launch_naive_attention(
    const T *Q,
    const T *K,
    const T *V,
    T *O,
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
) 
{
    T sm_scale = 1.0 / sqrtf(static_cast<T>(head_dim));
    
    // sm allocation
    // sm to store softmax output
    thrust::device_vector<T> d_sm(batch_size * num_heads * seq_len * seq_len);
    T* d_sm_ptr = d_sm.data().get();

    // sm = QK^T
    dim3 qk_grid(batch_size * num_heads, 1, 1);
    dim3 qk_block(seq_len , 1, 1);
    naive_gemm<T><<<qk_grid, qk_block, 0, stream>>>(Q, K, d_sm_ptr, sm_scale, 0, seq_len, seq_len, head_dim);
    cudaStreamSynchronize(stream);

    // softmax
    dim3 sm_grid(batch_size * num_heads, 1, 1);
    dim3 sm_block(seq_len, 1, 1);
    naive_softmax<T><<<sm_grid, sm_block, 0, stream>>>(d_sm_ptr, d_sm_ptr, seq_len);
    cudaStreamSynchronize(stream);

    // O = sm * V
    dim3 o_grid(batch_size * num_heads, 1, 1);
    dim3 o_block(seq_len, 1, 1);
    naive_pv<T><<<o_grid, o_block, 0, stream>>>(d_sm_ptr, V, O, seq_len, head_dim);
    cudaStreamSynchronize(stream);
}

// explicit instantiation
template void launch_naive_attention<float>
(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    unsigned  batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
);