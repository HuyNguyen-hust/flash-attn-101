#pragma once

#include <cute/tensor.hpp>

#include "utils.h"

namespace flash
{

using namespace cute;

template<bool zero_init, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__forceinline__ __device__ void thread_reduce(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &summary, Operator &op)
{
    // thread_reduce is for reducing within a thread, this do the reduction on (2*MMA_M) * (2*MMA_N) elements over dimension N --> return (2*MMA_M)

    static_assert(decltype(rank(tensor))::value == 2);
    static_assert(decltype(rank(summary))::value == 1);
    CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(summary));
    
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi)
    {   
        summary(mi) = zero_init ? tensor(mi, 0) : op(tensor(mi, 0), summary(mi));
        #pragma unroll
        for (int mj = 1; mj < size<1>(tensor); ++mj)
        {
            summary(mi) = op(tensor(mi, mj), summary(mi));
        }
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__forceinline__ __device__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op)
{   
    // quad_allreduce_ is for reducing across threads, this does the reduction over 4 threads
    // Why 4? Look at the MMA Atom you will see each row consists of 4 threads handling 8 elements
    // After thread_reduce, each thread do the reduction on (2*MMA_M) * (2*MMA_N) elements over dimension N and get (2*MMA_M)
    // And each row we have 4 threads doing so --> (2*MMA_M) * 4
    // This quad_allreduce_ do the reduction over 4 threads --> (2*MMA_M) in the end

    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__forceinline__ __device__ void reduce_(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &summary, Operator &op)
{
    thread_reduce<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void reduce_max(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max)
{
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void reduce_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &sum)
{
    SumOp<float> sum_op;
    thread_reduce<zero_init>(tensor, sum, sum_op);
}

template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, const float scale)
{
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

template <int kNRows> // 2 * MMA_M, number of rows that each thread is responsible for inside 1 tile
struct Softmax
{   
    using TensorT = decltype(
        make_tensor<float>(
            make_shape(Int<kNRows>{})
        )
    );
    TensorT row_max, row_sum;

    __device__ Softmax() {}

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, const float softmax_scale_log2)
    {   
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to ((2, MMA_M), (2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows); // 2 * MMA_M
        if (Is_first)
        {
            // m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
            flash::reduce_max</*zero_init=*/true>(scores, row_max);
            // P_i^(j) = exp(S_i^(j) - m_i^(j))
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // l_i^(j) = e^{m_i^(j-1) - m_i^(j)} * l_i^(j-1) + rowsum(P_i^(j))
            flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        }
        else
        {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);

            // m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
            flash::reduce_max</*zero_init=*/false>(scores, row_max);
            
            // Reshape acc_o from (MMA=4, MMA_M, MMA_N) to ((2, MMA_M), (2, MMA_N))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows); // 2 * MMA_M

            #pragma unroll
            for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi)
            {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2); // exp(m_i^(j-1) - m_i^(j))
                row_sum(mi) *= scores_scale;
                
                // pragma unroll
                for (int nj = 0; nj < size<1>(acc_o_rowcol); ++nj)
                {
                    acc_o_rowcol(mi, nj) *= scores_scale;
                }
            }

            // P_i^(j) = exp(S_i^(j) - m_i^(j))
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // l_i^(j) = e^{m_i^(j-1) - m_i^(j)} * l_i^(j-1) + rowsum(P_i^(j))
            flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
            // still need one quad_allreduce_ to do reduce between threads
        }
    }

    template<typename Tensor0>
    __forceinline__ __device__ void normalize_softmax_lse(Tensor0 &acc_o) 
    {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f) ? 1.f : 1.f / sum;
            float scale = inv_sum;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
    }

};

} // end namespace flash