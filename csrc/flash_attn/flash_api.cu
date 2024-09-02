#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"

#include "cuda_attn.hpp"

// set_params_fprop
template <typename T>
void set_params_fprop(
    Flash_fwd_params &params,
    const size_t b,
    const size_t seqlen_q, const size_t seqlen_k,
    const size_t h,
    const size_t d,
    const T *q,
    const T *k,
    const T *v,
    T *out,
    float softmax_scale,
    bool is_causal
)
{
    // reset params
    params = {};
    params.is_bf16 = false;

    // set the pointers and stride
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;
    params.o_ptr = out;

    params.q_batch_stride = h * seqlen_q * d;               // num_heads * seqlen_q * head_size
    params.q_head_stride  = seqlen_q * d;                   // seqlen_q * head_size
    params.q_row_stride   = d;                              // head_size

    params.k_batch_stride = h * seqlen_k * d;               // num_heads * seqlen_k * head_size
    params.k_head_stride  = seqlen_k * d;                   // seqlen_k * head_size
    params.k_row_stride   = d;                              // head_size

    params.v_batch_stride = h * seqlen_k * d;               // num_heads * seqlen_k * head_size
    params.v_head_stride  = seqlen_k * d;                   // seqlen_k * head_size
    params.v_row_stride   = d;                              // head_size

    params.o_batch_stride = h * seqlen_q * d;               // num_heads * seqlen_q * head_size
    params.o_head_stride  = seqlen_q * d;                   // seqlen_q * head_size
    params.o_row_stride   = d;                              // head_size

    // softmax scale
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // set the dimensions
    params.b = b;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.h = h;
    params.d = d;

    // causal
    params.is_causal = is_causal;
}

// run_mha_fwd
void run_mha_fwd(
    Flash_fwd_params &params,
    cudaStream_t stream
)
{
    HEADDIM_SWITCH(params.d, [&] {
        // These switches are to specify the template parameters
        // I hardcode the element_type and Is_causal here instead of using switches
        using elem_type = cutlass::half_t;
        constexpr bool Is_causal = true;
        run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
    });
}

// mha_fwd
template <typename T>
void mha_fwd
(
    const T *q,
    const T *k,
    const T *v,
    T *o,
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
)
{   
    // args shape
    // q: [batch_size, num_heads, seqlen_q, head_size]
    // k: [batch_size, num_heads, seqlen_k, head_size]
    // v: [batch_size, num_heads, seqlen_k, head_size]
    // o: [batch_size, num_heads, seqlen_q, head_size]
    // simplified version: same num_heads (no MQA/GQA)

    const int seqlen_q = seq_len;
    const int seqlen_k = seq_len;
    T softmax_scale = 1.0f / sqrtf(head_dim * 1.0f);
    bool is_causal = true;

    // set params
    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size, seqlen_q, seqlen_k, num_heads, head_dim,
        q, k, v,
        o,
        softmax_scale,
        is_causal
    );

    // call run_mha_fwd
    run_mha_fwd(params, stream);
}

// explicit instantiation
template void mha_fwd<__half>
(
    const __half *q,
    const __half *k,
    const __half *v,
    __half *o,
    unsigned int batch_size, unsigned int num_heads, unsigned int seq_len, unsigned int head_dim,
    cudaStream_t stream
);