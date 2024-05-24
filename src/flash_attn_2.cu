#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cassert>

#include <helper.cuh>

// helper function
void print_device_tensor(float* matrix, int m, int n);
void print_host_tensor(float* matrix, int m, int n);
bool all_close(float *A, float *B, int batch_size, int num_heads, int seq_len, int head_dim);

// kernel function
__global__ void flash_attention_2_kernel(
    float* Q, float* K, float* V, float* O,
    float* L,
    int Tr, int Tc, int Br, int Bc,
    int num_heads, int seq_len, int head_dim,
    float softmax_scale
);

// attn function
void flash_attn_2(float* d_Q, float* d_K, float* d_V, float* d_O,
    int batch_size, int num_heads, int seq_len, int head_dim);

__global__ void flash_attention_2_kernel(
    float* Q, float* K, float* V, float* O,
    float* L,
    int Tr, int Tc, int Br, int Bc,
    int num_heads, int seq_len, int head_dim,
    float softmax_scale
) {
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int block_id = blockIdx.z;

    int QKV_offset = batch_id * (num_heads * seq_len * head_dim) + head_id * (seq_len * head_dim);
    int L_offset = batch_id * (num_heads * seq_len) + head_id * (seq_len);

    extern __shared__ float sram[];
    int tile_size = Br * head_dim;  // size of Qi, Kj, Vj
    float* s_Qi = sram;
    float* s_Kj = &sram[tile_size];
    float* s_Vj = &sram[tile_size * 2];
    float* s_S = &sram[tile_size * 3];

    
    if (block_id * Br + threadIdx.x >= seq_len)
        return;
    
    // load Qi from HBM to SRAM
    int Qi_offset = QKV_offset + block_id * Br * head_dim;
    for (int d = 0; d < head_dim; d++) {
        s_Qi[threadIdx.x * head_dim + d] = Q[Qi_offset + threadIdx.x * head_dim + d];
    }

    // li, mi initialization

    float li = 0.0f;
    float mi = -INFINITY;
    float last_mi;
    float last_li;

    // for loop over Tc
    for (int j = 0; j <= block_id; j++) {
        __syncthreads();
        // load Kj and Vj from HBM to SRAM
        int Kj_offset = QKV_offset + j * Bc * head_dim;
        int Vj_offset = QKV_offset + j * Bc * head_dim;
        for (int d = 0; d < head_dim; d++) {
            s_Kj[threadIdx.x * head_dim + d] = K[Kj_offset + threadIdx.x * head_dim + d];
            s_Vj[threadIdx.x * head_dim + d] = V[Vj_offset + threadIdx.x * head_dim + d];
        }
        __syncthreads();

        // Sij = Qi @ Kj^T
        // last_mi = mi
        // mi = max(last_mi, rowmax(Sij))
        last_mi = mi;
        float rowmax = -INFINITY;
        for (int k = 0; k < Bc; k++) {
            if (j * Bc + k >= seq_len) {
                break;
            }
            if (j * Bc + k > block_id * Br + threadIdx.x) {
                break;
            }
            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum += s_Qi[threadIdx.x * head_dim + d] * s_Kj[k * head_dim + d];
            }
            sum *= softmax_scale;
            s_S[threadIdx.x * Bc + k] = sum;
            rowmax = rowmax > sum ? rowmax : sum;
        }
        mi = last_mi > rowmax ? last_mi : rowmax;

        // P_ij = exp(Sij - mi), s_S <-- P_ij
        // rowsum(Pij)
        float rowsum = 0.0f;
        for (int k = 0; k < Bc; k++) {
            if (j * Bc + k >= seq_len) {
                break;
            }
            if (j * Bc + k > block_id * Br + threadIdx.x) {
                break;
            }
            else {
                s_S[threadIdx.x * Bc + k] = __expf(s_S[threadIdx.x * Bc + k] - mi);
            }
            rowsum += s_S[threadIdx.x * Bc + k];
        }
        
        // alpha = exp(last_mi - mi)
        // li = alpha * last_li + rowsum
        last_li = li;
        float alpha = __expf(last_mi - mi);
        li = alpha * last_li + rowsum;

        // Oi = (1 / alpha) * last_Oi + Pij * Vj
        int Oi_offset = QKV_offset + block_id * Br * head_dim;
        for (int d = 0; d < head_dim; d++) {
            // pv = Pij * Vj
            float pv = 0.0f;
            for (int k = 0; k < Bc; k++) {
                if (j * Bc + k >= seq_len) {
                    break;
                }
                if (j * Bc + k > block_id * Br + threadIdx.x) {
                    break;
                }
                else {
                    pv += s_S[threadIdx.x * Bc + k] * s_Vj[k * head_dim + d];
                }
            }
            O[Oi_offset + threadIdx.x * head_dim + d] = (1.0f / alpha) * O[Oi_offset + threadIdx.x * head_dim] + pv;
        }
    }
    // Oi = (1 / li) * Oi
    // Li = mi + log(li)
    // store Oi to HBM
    int Oi_offset = QKV_offset + block_id * Br * head_dim;
    for (int d = 0; d < head_dim; d++) {
        O[Oi_offset + threadIdx.x * head_dim + d] /= li;
    }
    // store Li to HBM
    int li_offset = L_offset + block_id * Br;
    L[li_offset + threadIdx.x] = mi + logf(li);
}

void flash_attn_2(float* Q, float* K, float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim) {
        // softmax_scale
        float softmax_scale = 1.0f / sqrtf(head_dim * 1.0f);
        
        // compute Tr, Tc
        assert(Br == Bc && "Only support Br == Bc");
        int Tr = seq_len / Br;
        int Tc = seq_len / Bc;

        float *d_L;
        cudaMalloc(&d_L, batch_size * seq_len * num_heads * sizeof(float));

        dim3 grid(batch_size, num_heads, Tr);
        dim3 block(Br);

        const int sram_size = (3 * Bc * head_dim * sizeof(float)) + (Bc * Br * sizeof(float));
        
        flash_attention_2_kernel<<<grid, block, sram_size>>>(
            Q, K, V, O,
            d_L,
            Tr, Tc, Br, Bc,
            num_heads, seq_len, head_dim,
            softmax_scale
        );
    }

void test_flash_attn_2() {
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
        O[i] = 0.0f;
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

    // execute operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    flash_attn_2(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for flash attention 2 kernel execution: %.3f ms \n", milliseconds / 100);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy to host
    cudaMemcpy(O, d_O, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // gpu free
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
//     test_flash_attn_2();
//     return 0;
// }