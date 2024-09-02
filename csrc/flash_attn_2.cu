#include <cmath>
#include <cassert>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>

#include "cuda_attn.hpp"

// kernel
template <typename T>
__global__ void flash_attention_2_kernel(
    const T* Q, const T* K, const T* V,
    T* O,
    T* L,
    unsigned int Tr, unsigned int Tc,
    unsigned int Br, unsigned int Bc,
    unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    T softmax_scale
) 
{
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int block_id = blockIdx.z;

    int QKV_offset = batch_id * (num_heads * seq_len * head_dim) + head_id * (seq_len * head_dim);
    int L_offset = batch_id * (num_heads * seq_len) + head_id * (seq_len);

    extern __shared__ T sram[];
    int tile_size = Br * head_dim;  // size of Qi, Kj, Vj
    T* s_Qi = sram;
    T* s_Kj = &sram[tile_size];
    T* s_Vj = &sram[tile_size * 2];
    T* s_S = &sram[tile_size * 3];

    
    if (block_id * Br + threadIdx.x >= seq_len)
        return;
    
    // load Qi from HBM to SRAM
    int Qi_offset = QKV_offset + block_id * Br * head_dim;
    for (int d = 0; d < head_dim; d++) {
        s_Qi[threadIdx.x * head_dim + d] = Q[Qi_offset + threadIdx.x * head_dim + d];
    }

    // li, mi initialization

    T li = static_cast<T>(0.0f);
    T mi = -INFINITY;
    T last_mi;
    T last_li;

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
        T rowmax = -INFINITY;
        for (int k = 0; k < Bc; k++) {
            if (j * Bc + k >= seq_len) {
                break;
            }
            if (j * Bc + k > block_id * Br + threadIdx.x) {
                break;
            }
            T sum = static_cast<T>(0.0f);
            for (int d = 0; d < head_dim; d++) {
                sum += s_Qi[threadIdx.x * head_dim + d] * s_Kj[k * head_dim + d];
            }
            sum *= softmax_scale;
            s_S[threadIdx.x * Bc + k] = sum;
            rowmax = rowmax > sum ? rowmax : sum;
        }
        last_mi = mi;
        mi = last_mi > rowmax ? last_mi : rowmax;

        // P_ij = exp(Sij - mi), s_S <-- P_ij
        // rowsum(Pij)
        T rowsum = static_cast<T>(0.0f);
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
        T alpha = __expf(last_mi - mi);
        li = alpha * last_li + rowsum;

        // Oi = alpha * last_Oi + Pij * Vj
        int Oi_offset = QKV_offset + block_id * Br * head_dim;
        for (int d = 0; d < head_dim; d++) {
            // pv = Pij * Vj
            T pv = static_cast<T>(0.0f);
            for (int k = 0; k < Bc; k++) {
                if (j * Bc + k >= seq_len) {
                    break;
                }
                if (j * Bc + k > block_id * Br + threadIdx.x) {
                    break;
                }
                pv += s_S[threadIdx.x * Bc + k] * s_Vj[k * head_dim + d];
            }
            O[Oi_offset + threadIdx.x * head_dim + d] = alpha * O[Oi_offset + threadIdx.x * head_dim + d] + pv;
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
    L[li_offset + threadIdx.x] = mi + static_cast<T>(logf(li));
}

// launch
template <typename T>
void launch_flash_attention_02(
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
    
    // compute Tr, Tc
    unsigned int Br = 16U;
    unsigned int Bc = 16U;
    assert(Br == Bc && "Only support Br == Bc");
    assert(seq_len % Br == 0 && "seq_len must be divisible by Br");
    int Tr = seq_len / Br;
    int Tc = seq_len / Bc;

    thrust::device_vector<T> d_L(batch_size * seq_len * num_heads);
    T* d_L_ptr = d_L.data().get();

    dim3 grid(batch_size, num_heads, Tr);
    dim3 block(Br);

    const int sram_size = (3 * Bc * head_dim * sizeof(T)) + (Bc * Br * sizeof(T));
    cudaFuncSetAttribute(flash_attention_2_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, sram_size);
    
    flash_attention_2_kernel<T><<<grid, block, sram_size>>>(
        Q, K, V, O,
        d_L_ptr,
        Tr, Tc, Br, Bc,
        num_heads, seq_len, head_dim,
        softmax_scale
    );
}

// explicit instantiation
// template void launch_flash_attention_02<float>
// (
//     const float *Q,
//     const float *K,
//     const float *V,
//     float *O,
//     unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
//     cudaStream_t stream
// );

template void launch_flash_attention_02<__half>
(
    const __half *Q,
    const __half *K,
    const __half *V,
    __half *O,
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
);