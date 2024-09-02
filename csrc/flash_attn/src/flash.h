#pragma once

#include <cuda.h>
#include <vector>

// Qkv_params
struct Qkv_params
{   
    // index_t is used in calculating offset
    using index_t = uint32_t;

    // qkv pointers
    const void* __restrict__ q_ptr;
    const void* __restrict__ k_ptr;
    const void* __restrict__ v_ptr;

    // strides
    // simplified version: same seqlen, same num_heads
    index_t q_batch_stride;
    index_t q_row_stride;
    index_t q_head_stride;
    index_t k_batch_stride;
    index_t k_row_stride;
    index_t k_head_stride;
    index_t v_batch_stride;
    index_t v_row_stride;
    index_t v_head_stride;
};

// Flash_fwd_params
struct Flash_fwd_params : public Qkv_params
{
    // o pointers
    void* __restrict__ o_ptr;
    void* __restrict__ oaccum_ptr;
    
    // strides
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;  

    // the dimensions
    int b, seqlen_q, seqlen_k, h, d;

    // scaling factor
    float scale_softmax;
    float scale_softmax_log2;
    
    // bf16, causal flags
    bool is_bf16;
    bool is_causal;
};

template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);