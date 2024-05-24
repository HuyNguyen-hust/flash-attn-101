#include <stdio.h>
#include <stdlib.h>

#include <helper.cuh>
#include <self_attention.cuh>
#include <flash_attn_1.cuh>
#include <flash_attn_2.cuh>

// helper function
void print_device_tensor(float* matrix, int m, int n);
void print_host_tensor(float* matrix, int m, int n);
bool all_close(float *A, float *B, int batch_size, int num_heads, int seq_len, int head_dim);

// attn function
void self_attention_cuda(float* Q, float* K, float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim);

void flash_attn_1(float* d_Q, float* d_K, float* d_V, float* d_O,
    int batch_size, int num_heads, int seq_len, int head_dim);

void flash_attn_2(float* d_Q, float* d_K, float* d_V, float* d_O,
    int batch_size, int num_heads, int seq_len, int head_dim);

// test function
template<typename Func>
float* test_attn(Func attn);

//bench function
void bench();

template<typename Func>
float* test_attn(Func attn) {
    // host allocation
    float* Q = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    float* K = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    float* V = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    float* O = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));

    // host initialization
    for (int i = 0; i < batch_size * num_heads * seq_len * head_dim; i++) {
        Q[i] = static_cast<float>(i) / 1024.0f;
        K[i] = static_cast<float>(i) / 1024.0f;
        V[i] = static_cast<float>(i) / 1024.0f;
    }

    // gpu allocation
    float* d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, batch_size * num_heads * seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_K, batch_size * num_heads * seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_V, batch_size * num_heads * seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_O, batch_size * num_heads * seq_len * head_dim * sizeof(float));

    // copy to gpu
    cudaMemcpy(d_Q, Q, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, O, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // execute operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    attn(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel execution: %.3f ms \n", milliseconds / 100);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy to host
    cudaMemcpy(O, d_O, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    free(Q);
    free(K);
    free(V);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return O;
}

void bench() {
    printf("----------- naive self attention -----------\n");
    float* O = test_attn(self_attention_cuda);
    
    printf("----------- flash attention 1 ----------- \n");
    float* O_1 = test_attn(flash_attn_1);
    
    printf("----------- flash attention 2 ----------- \n");
    float* O_2 = test_attn(flash_attn_2);
    
    printf("----------- sanity check ----------- \n");
    printf("all close: %d\n", all_close(O, O_1, batch_size, num_heads, seq_len, head_dim)
        && all_close(O, O_2, batch_size, num_heads, seq_len, head_dim));
}


int main() {
    bench();
    return 0;
}