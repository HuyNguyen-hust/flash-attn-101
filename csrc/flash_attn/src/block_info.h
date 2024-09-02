#pragma once

#include <cuda_runtime.h>

namespace flash
{
struct BlockInfo
{
    // Constructor
    template <typename Params>
    __device__ BlockInfo(const Params &params)
    : seqlen_q(params.seqlen_q)
    , seqlen_k(params.seqlen_k)
    {
    }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t head_stride, const int bidb, const int bidh) const
    {
        return batch_stride * bidb + head_stride * bidh;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t head_stride, const int bidb, const int bidh) const
    {
        return batch_stride * bidb + head_stride * bidh;
    }

    // Sequence length for Q
    const int seqlen_q;
    // Sequence length for K
    const int seqlen_k;
};

} // end namespace flash
