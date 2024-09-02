#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace flash
{

using namespace cute;

template<
    typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4,
    typename TiledMMA, typename TiledCopyA, typename TiledCopyB, typename ThrCopyA, typename ThrCopyB
>
__forceinline__ __device__ void gemm(
    Tensor0 &acc,
    Tensor1 &tCrA, Tensor2 &tCrB,
    Tensor3 &tCsA, Tensor4 &tCsB,
    TiledMMA tiled_mma,
    TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B
)
{   
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K

    // S2R copy A, B
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA); // (CPY,CPY_M,CPY_K)
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB); // (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N

    // Intra-tile pipeline
    // Prefetch k=0
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));

    // Iterate over MMA_K
    auto MMA_K = size<2>(tCrA);
    for (int k = 0; k < MMA_K; ++k) {
        if (k < MMA_K - 1) {
            // Prefetch the next k
            cute::copy(smem_tiled_copy_A, tCsA(_, _, k + 1), tCrA_copy_view(_, _, k + 1));
            cute::copy(smem_tiled_copy_B, tCsB(_, _, k + 1), tCrB_copy_view(_, _, k + 1));
        }

        // MMA
        cute::gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), acc);
    }
}


template<
    typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
    typename TiledMMA, typename TiledCopyB, typename ThrCopyB
>
__forceinline__ __device__ void gemm_rs(
    Tensor0 &acc,
    Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 &tCsB,
    TiledMMA tiled_mma,
    TiledCopyB smem_tiled_copy_B, 
    ThrCopyB smem_thr_copy_B
)
{
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K

    // S2R copy B
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB); // (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N

    // Intra-tile pipeline
    // Prefetch k=0
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));

    // Iterate over MMA_K
    auto MMA_K = size<2>(tCrA);
    for (int k = 0; k < MMA_K; ++k) {
        if (k < MMA_K - 1) {
            // Prefetch the next k
            cute::copy(smem_tiled_copy_B, tCsB(_, _, k + 1), tCrB_copy_view(_, _, k + 1));
        }

        // MMA
        cute::gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), acc);
    }
}

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout)
{
    // acc_layout shape: (MMA=4, MMA_M, MMA_N)
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    
    Layout l = logical_divide(acc_layout, Shape<_2>{}); 
    // This is just the logical divide on the first mode of the acc_layout. See notebook/convert_layout_acc_rowcol.cu
    // ((2, 2), MMA_M, MMA_N)

    return make_layout(
        make_layout(get<0, 1>(l), get<1>(l)),   // (2, MMA_M),
        make_layout(get<0, 0>(l), get<2>(l))    // (2, MMA_N),
    );
    // The 4 elements in the original layout are first arranged as:
    // 0 1 2 3

    // After logical_divide, we have ((2, 2), MMA_M, MMA_N)
    // The first '2' represents the number of elements that one thread takes care in one row    --> get<0, 0>(l) 
    // The second '2' represents the two rows (top row 0-1, bottom row 2-3)                     --> get<0, 1>(l)

    // Therefore:
    // make_layout(get<0, 1>(l), get<1>(l)) creates (2, MMA_M) for rows
    // make_layout(get<0, 0>(l), get<2>(l)) creates (2, MMA_N) for columns
}

// Reshape rP from (MMA=4, MMA_M, MMA_N) 
// to ((4, 2), MMA_M, MMA_N/2) if using m16n8k16
// or (4, MMA_M, MMA_N) if using m16n8k8
// Why do we need to reshape?
// Because other thread partitioning like tOrVt are done by tiled_mma
// However rP is the output of gemm 1 and of shape (MMA=4, MMA_M, MMA_N) which is not compatible with tiled_mma
// see the A layout of m16n8k16 in notebook/m16n8k16.cu
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout)
{
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) 
    {
        return acc_layout;
    }
    else 
    {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }

    // Why does this reshape (using logical_divide) work?
    // take the thread 0 as an example
    // See the C (rP before reshaping) layout of m16n8k16, thread 0 is now responsible for 4 elements at (0,0) (0,1) (8,0) (8,1)
    // See the A (rP shape needed for tiled_mma) layout of m16n8k16, thread 0 is now responsible for (4,2) elememts (0,0) (0,1) (8,0) (8,1) (0, 8) (0, 9) (8, 8) (8, 9)
    // Luckily they are aligned in the layout, so we only need to double the size of MMA in the N dimension to get (4,2)
    // However we also have to halve MMA_N
}

template<int THREADS>
struct Allreduce 
{
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) 
    {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> 
{
    template<typename T, typename Operator> 
    static __device__ __forceinline__ T run(T x, Operator &op) 
    {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) 
{
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous" ?
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() 
{
    #if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
        asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
    #endif
}

template<typename T>
struct MaxOp 
{
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> 
{
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

template<typename T>
struct SumOp 
{
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

} // end namespace flash