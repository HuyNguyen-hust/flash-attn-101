#include <cmath>
#include <cassert>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>

#include "cuda_attn.hpp"

// kernel
template <typename T>
__global__ void flash_attention_1_kernel(
    const T* Q, const T* K, const T* V, T* O,
    T* l, T* m,
    unsigned int Tr, unsigned int Tc, 
    unsigned int Br, unsigned int Bc, 
    unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    T softmax_scale
) 
{
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;

    int QKV_offset = batch_id * (num_heads * seq_len * head_dim) + head_id * (seq_len * head_dim);
    int lm_offset = batch_id * (num_heads * seq_len) + head_id * (seq_len);

    extern __shared__ T sram[];
    int tile_size = Br * head_dim;  // size of Qi, Kj, Vj
    T* s_Qi = sram;
    T* s_Kj = &sram[tile_size];
    T* s_Vj = &sram[tile_size * 2];
    T* s_S = &sram[tile_size * 3];

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
            T li = l[li_offset + threadIdx.x];
            T mi = m[mi_offset + threadIdx.x];

            // Sij = Qi @ Kj^T
            // cur_m_ij = row_max(Sij)        
            // mi_new = max(mi, cur_m_ij)
            T cur_m_ij = -INFINITY;    
            for (int k = 0; k < Bc; k++) {
                if (j * Bc + k >= seq_len)
                    break;
                T sum = static_cast<T>(0.0f);
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
            T mi_new = mi > cur_m_ij ? mi : cur_m_ij;

            // P_ij = __expf(Sij - cur_m_ij) , s_S <-- P_ij
            // cur_l_ij = row_sum(P_ij)
            // li_new = __expf(mi-mi_new) * li + __expf(cur_m_ij - mi_new) * cur_l_ij
            T cur_l_ij = static_cast<T>(0.0f);
            for (int k = 0; k < Bc; k++) {
                if (j * Bc + k >= seq_len)
                    break;
                if (i * Br + threadIdx.x < j * Bc + k) {
                    s_S[threadIdx.x * Bc + k] = static_cast<T>(0.0f);
                } else {
                    s_S[threadIdx.x * Bc + k] = __expf(s_S[threadIdx.x * Bc + k] - cur_m_ij);
                }
                cur_l_ij += s_S[threadIdx.x * Bc + k];                
            }
            T li_new = __expf(mi - mi_new) * li + __expf(cur_m_ij - mi_new) * cur_l_ij;
            
            // alpha = __expf(mi-mi_new)
            // beta = __expf(cur_m_ij - mi_new)
            // Oi = 1 / li_new * (li * alpha * Oi + beta * Pij * Vj)
            T alpha = __expf(mi - mi_new);
            T beta = __expf(cur_m_ij - mi_new);
            
            int Oi_offset = QKV_offset + i * Br * head_dim;

            for (int d = 0; d < head_dim; d++) {
                // for loop over Bc
                T pv = static_cast<T>(0.0f);
                for (int k = 0; k < Bc; k++) {
                    if (j * Bc + k >= seq_len)
                    {
                        break;
                    }
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

// launch
template <typename T>
void launch_flash_attention_01(
    const T *Q,
    const T *K,
    const T *V,
    T *O,
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
) 
{
    // softmax_scale
    T softmax_scale = 1.0f / sqrtf(head_dim * 1.0f);
    //  compute Tr, Tc
    unsigned int Br = 16U;
    unsigned int Bc = 16U;
    assert(Br == Bc && "Only support Br == Bc");
    assert(seq_len % Br == 0 && "seq_len must be divisible by Br");
    int Tr = seq_len / Br;
    int Tc = seq_len / Bc;

    // Initialize l, m
    thrust::device_vector<T> d_l(batch_size * num_heads * seq_len);
    thrust::device_vector<T> d_m(batch_size * num_heads * seq_len);

    thrust::fill(d_l.begin(), d_l.end(), static_cast<T>(0.0f));
    thrust::fill(d_m.begin(), d_m.end(), static_cast<T>(-INFINITY));

    T* d_l_ptr = d_l.data().get();
    T* d_m_ptr = d_m.data().get();

    dim3 grid(batch_size, num_heads);
    dim3 block(Br);
    
    const int sram_size = (3 * Bc * head_dim * sizeof(T)) + (Bc * Br * sizeof(T));
    cudaFuncSetAttribute(flash_attention_1_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, sram_size);

    flash_attention_1_kernel<T><<<grid, block, sram_size>>>(
        Q, K, V, O,
        d_l_ptr, d_m_ptr, 
        Tr, Tc, Br, Bc, 
        num_heads, seq_len, head_dim,
        softmax_scale
    );
}

// explicit instantiation
template void launch_flash_attention_01<float>
(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
);