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
__global__ void flash_attention_1_kernel(
    float* Q, float* K, float* V, float* O,
    float* l, float* m,
    int Tr, int Tc, int Br, int Bc,
    int num_heads, int seq_len, int head_dim,
    float softmax_scale
);

// attn function
void flash_attn_1(float* d_Q, float* d_K, float* d_V, float* d_O,
    int batch_size, int num_heads, int seq_len, int head_dim);

__global__ void flash_attention_1_kernel(
    float* Q, float* K, float* V, float* O,
    float* l, float* m,
    int Tr, int Tc, int Br, int Bc, 
    int num_heads, int seq_len, int head_dim,
    float softmax_scale
) {
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;

    int QKV_offset = batch_id * (num_heads * seq_len * head_dim) + head_id * (seq_len * head_dim);
    int lm_offset = batch_id * (num_heads * seq_len) + head_id * (seq_len);

    extern __shared__ float sram[];
    int tile_size = Br * head_dim;  // size of Qi, Kj, Vj
    float* s_Qi = sram;
    float* s_Kj = &sram[tile_size];
    float* s_Vj = &sram[tile_size * 2];
    float* s_S = &sram[tile_size * 3];

    // for loop over Tc
    for (int j = 0; j < Tc; j++) {
        int Kj_offset = QKV_offset + j * Bc * head_dim;
        int Vj_offset = QKV_offset + j * Bc * head_dim;
        // each thread loads a row of Kj and Vj
        for (int d = 0; d < head_dim; d++) {
            s_Kj[threadIdx.x * head_dim + d] = K[Kj_offset + threadIdx.x * head_dim + d];
            s_Vj[threadIdx.x * head_dim + d] = V[Vj_offset + threadIdx.x * head_dim + d];
        }
        __syncthreads();
        
        // for loop over Tr
        for (int i = j; i < Tr; i++) {
            if (i * Br + threadIdx.x >= seq_len)
                break;
            int Qi_offset = QKV_offset + i * Br * head_dim;
            // each thread loads a row of Qi
            for (int d = 0; d < head_dim; d ++) {
                s_Qi[threadIdx.x * head_dim + d] = Q[Qi_offset + threadIdx.x * head_dim + d];
            }

            int li_offset = lm_offset + i * Br;
            int mi_offset = lm_offset + i * Br;
            // each thread loads a value of l and m
            float li = l[li_offset + threadIdx.x];
            float mi = m[mi_offset + threadIdx.x];

            // Sij = Qi @ Kj^T
            // cur_m_ij = row_max(Sij)        
            // mi_new = max(mi, cur_m_ij)
            float cur_m_ij = -INFINITY;    
            for (int k = 0; k < Bc; k++) {
                if (j * Bc + k >= seq_len)
                    break;
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    sum += s_Qi[threadIdx.x * head_dim + d] * s_Kj[k * head_dim + d];
                }
                sum *= softmax_scale;
                if (i * Br + threadIdx.x < j * Bc + k) { // threadIdx.x < k is not enough
                    sum = -INFINITY;
                }
                s_S[threadIdx.x * Bc + k] = sum;
                cur_m_ij = cur_m_ij > sum ? cur_m_ij : sum;
            }
            float mi_new = mi > cur_m_ij ? mi : cur_m_ij;

            // P_ij = __expf(Sij - cur_m_ij) , s_S <-- P_ij
            // cur_l_ij = row_sum(P_ij)
            // li_new = __expf(mi-mi_new) * li + __expf(cur_m_ij - mi_new) * cur_l_ij
            float cur_l_ij = 0.0f;
            for (int k = 0; k < Bc; k++) {
                if (j * Bc + k >= seq_len)
                    break;
                if (i * Br + threadIdx.x < j * Bc + k) {
                    s_S[threadIdx.x * Bc + k] = 0.0f;
                } else {
                    s_S[threadIdx.x * Bc + k] = __expf(s_S[threadIdx.x * Bc + k] - cur_m_ij);
                }
                cur_l_ij += s_S[threadIdx.x * Bc + k];                
            }
            float li_new = __expf(mi - mi_new) * li + __expf(cur_m_ij - mi_new) * cur_l_ij;
            
            // alpha = __expf(mi-mi_new)
            // beta = __expf(cur_m_ij - mi_new)
            // Oi = 1 / li_new * (li * alpha * Oi + beta * Pij * Vj)
            float alpha = __expf(mi - mi_new);
            float beta = __expf(cur_m_ij - mi_new);
            
            int Oi_offset = QKV_offset + i * Br * head_dim;

            for (int d = 0; d < head_dim; d++) {
                // for loop over Bc
                float pv = 0.0f;
                for (int k = 0; k < Bc; k++) {
                    pv += s_S[threadIdx.x * Bc + k] * s_Vj[k * head_dim + d];
                }
            
                O[Oi_offset + threadIdx.x * head_dim + d] = (1.0f / li_new) \
                * (li * alpha * O[Oi_offset + threadIdx.x * head_dim + d] \
                + beta * pv);
            }

            // li = li_new
            l[li_offset + threadIdx.x] = li_new;
            // mi = mi_new
            m[mi_offset + threadIdx.x] = mi_new;
        }
        __syncthreads(); 
        // there might be cases where one thread finishes the inner loop 
        // and increases i by 1, which means the next Kj and Vj will be loaded
        // while other threads are still in the inner loop, and use the wrong Kj and Vj.
    }
}

void flash_attn_1(float* d_Q, float* d_K, float* d_V, float* d_O,
    int batch_size, int num_heads, int seq_len, int head_dim) {
        // softmax_scale
        float softmax_scale = 1.0f / sqrt(head_dim * 1.0f);
        //  compute Tr, Tc
        assert(Br == Bc && "Only support Br == Bc");
        int Tr = seq_len / Br;
        int Tc = seq_len / Bc;

        // Initialize l, m
        float *l, *m;
        l = (float*)malloc(batch_size * seq_len * num_heads * sizeof(float));
        m = (float*)malloc(batch_size * seq_len * num_heads * sizeof(float));

        for (int i = 0; i < batch_size * seq_len * num_heads; i++) {
            l[i] = 0.0f;
            m[i] = -INFINITY;
        }

        float *d_l, *d_m;
        cudaMalloc(&d_l, batch_size * seq_len * num_heads * sizeof(float));
        cudaMalloc(&d_m, batch_size * seq_len * num_heads * sizeof(float));

        cudaMemcpy(d_l, l, batch_size * seq_len * num_heads * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, m, batch_size * seq_len * num_heads * sizeof(float), cudaMemcpyHostToDevice);

        dim3 grid(batch_size, num_heads);
        dim3 block(Br);
        
        const int sram_size = (3 * Bc * head_dim * sizeof(float)) + (Bc * Br * sizeof(float));

        flash_attention_1_kernel<<<grid, block, sram_size>>>(d_Q, d_K, d_V, d_O,
            d_l, d_m, 
            Tr, Tc, Br, Bc, 
            num_heads, seq_len, head_dim,
            softmax_scale
        );
    }
    
void test_flash_attn_1() {
    // host allocation 
    float* Q = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    float* K = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    float* V = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    float* O = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));
    float* O_naive = (float*)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));

    // initialization
    for (int i = 0; i < batch_size * num_heads * seq_len * head_dim; i++) {
        Q[i] = static_cast<float>(i) / 1024.0f;
        K[i] = static_cast<float>(i) / 1024.0f;
        V[i] = static_cast<float>(i) / 1024.0f;
        O[i] = 0.0f;
        O_naive[i] = 0.0f;
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

    flash_attn_1(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for flash attention 1 kernel execution: %.3f ms \n", milliseconds / 100);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy from gpu
    cudaMemcpy(O, d_O, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    free(Q);
    free(K);
    free(V);
    free(O);
}

// int main() {
//     test_flash_attn_1();
//     return 0;
// }