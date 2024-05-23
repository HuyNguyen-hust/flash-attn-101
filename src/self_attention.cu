#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cassert>

#include <include/helper.cuh>

// helper function
void print_device_tensor(float* matrix, int m, int n);
void print_host_tensor(float* matrix, int m, int n);
bool all_close(float *A, float *B, int batch_size, int num_heads, int seq_len, int head_dim);

// kernel function
__global__ void naive_gemm(float *A, float *B, float *C, 
                                float a, float b,
                                int M, int N, int K, 
                                int num_blocks);
__global__ void naive_softmax(float *input, float *output, int n);
__global__ void naive_pv(float *P, float *V, float *O, int M, int N);

// attn function
void self_attention_cuda(float *Q, float *K, float *V, float *O,
    int batch_size, int num_heads, int seq_len, int head_dim);

// gemm: aA@B + bC, A (m, k), B(n, k), C(m, n)
// m: seq_len, n: seq_len, k: head_dim
__global__ void naive_gemm(
    float *A, float *B, float *C, float a, float b,
    int M, int N, int K
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int j = 0; j < N; j++) {
        if (j > threadIdx.x) {
            C[idx * N + j] = -INFINITY;
        }
        else {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += A[idx * K + k] * B[(blockDim.x * blockIdx.x + j) * K + k];
            }
            C[idx * N + j] = a * sum + b * C[idx * N + j];
        }
    }
}

// softmax: exp(x - max(x)) / sum(exp(x - max(x)))
__global__ void naive_softmax(float *input, float *output, int N) {
    // 3-pass softmax
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float row_max = -INFINITY;
    float sum = 0.f;

    // max
    for (int j = 0; j < N; j++) {
        row_max = row_max > input[idx * N + j] ? row_max : input[idx * N + j];
    }  

    // sum
    for (int j = 0; j < N; j++) {
        if (j > threadIdx.x) {
            output[idx * N + j] = 0.f;
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
__global__ void naive_pv(
    float *P, float *V, float *O,
    int M, int N
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int j = 0; j < N; j++) {
        float sum = 0.f;
        for (int m = 0; m < M; m++) {
            sum += P[idx * M + m] * V[(blockDim.x * blockIdx.x + m) * N + j];
        }
        O[idx * N + j] = sum;
    }
}

void self_attention_cuda(float *Q, float *K, float *V, float *O, 
    int batch_size, int num_heads, int seq_len, int head_dim) {

    float sm_scale = 1.0 / sqrtf(static_cast<float>(head_dim));
    
    // sm allocation
    // sm to store softmax output
    float* d_sm;
    cudaMalloc(&d_sm, batch_size * num_heads * seq_len * seq_len * sizeof(float));

    // sm = QK^T
    dim3 qk_grid(batch_size * num_heads, 1, 1);
    dim3 qk_block(seq_len , 1, 1);
    naive_gemm<<<qk_grid, qk_block>>>(Q, K, d_sm, sm_scale, 0, seq_len, seq_len, head_dim);
    cudaDeviceSynchronize();

    // softmax
    dim3 sm_grid(batch_size * num_heads, 1, 1);
    dim3 sm_block(seq_len, 1, 1);
    naive_softmax<<<sm_grid, sm_block>>>(d_sm, d_sm, seq_len);
    cudaDeviceSynchronize();

    // O = sm * V
    dim3 o_grid(batch_size * num_heads, 1, 1);
    dim3 o_block(seq_len, 1, 1);
    naive_pv<<<o_grid, o_block>>>(d_sm, V, O, seq_len, head_dim);
    cudaDeviceSynchronize();

    // free sm
    cudaFree(d_sm);
}

void test_attention() {
    // host allocation
    float* Q = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    float* K = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    float* V = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    float* O = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));

    // host intialization
    for (int i = 0; i < batch_size * num_heads * seq_len * head_dim; i++) {
        Q[i] = static_cast<float>(i) / 1024.0f;
        K[i] = static_cast<float>(i) / 1024.0f;
        V[i] = static_cast<float>(i) / 1024.0f;
    }

    // device allocation
    float* d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, batch_size * num_heads * seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_K, batch_size * num_heads * seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_V, batch_size * num_heads * seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_O, batch_size * num_heads * seq_len * head_dim * sizeof(float));

    // copy to device
    cudaMemcpy(d_Q, Q, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);

    // execute operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    self_attention_cuda(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel execution: %.3f ms \n", milliseconds / 100);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy from device
    cudaMemcpy(O, d_O, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    // device free
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    // host free
    free(Q);
    free(K);
    free(V);
    free(O);
}

// int main() {
//     test_attention();
//     return 0;
// }
