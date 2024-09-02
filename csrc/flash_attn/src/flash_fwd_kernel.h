#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "utils.h"
#include "kernel_traits.h"
#include "mask.h"
#include "softmax.h"

namespace flash
{

using namespace cute;

template<typename Kernel_traits, bool Is_causal, typename Params>
inline __device__ void compute_attn_1rowblock(const Params params, const int bidb, const int bidh, const int m_block)
{
    // thread index
    const int tidx = threadIdx.x;

    // precision
    using Element = typename Kernel_traits::Element;
    using ElememntAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // shared memory
    extern __shared__ char smem_[];

    // kBlockM, kBlockN, kHeadDim, kNWarps
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    // BlockInfo: This class is later used for calculating offset
    BlockInfo binfo(params);

    if (m_block * kBlockM >= params.seqlen_q) return;

    // n_block_min, n_block_max
    const int n_block_min = 0;
    const int n_block_max = std::min(cute::ceil_div(params.seqlen_k, kBlockN), cute::ceil_div((m_block+1) * kBlockM, kBlockN));

    // tensors mQ, mK, mV
    // shape
    // mQ: [seqlen_q, d]
    // mK: [seqlen_k, d]
    // mV: [seqlen_k, d]
    // all in row-major
    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_head_stride, bidb, bidh)),
        make_shape(params.seqlen_q, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{})
    );
    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Element*>(params.k_ptr) + binfo.k_offset(params.k_batch_stride, params.k_head_stride, bidb, bidh)),
        make_shape(params.seqlen_k, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{})
    );
    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Element*>(params.v_ptr) + binfo.k_offset(params.v_batch_stride, params.v_head_stride, bidb, bidh)),
        make_shape(params.seqlen_k, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{})
    );

    // tiling to get gQ, gK, gV
    // shape
    // gQ: [kBlockM, kHeadDim, nblocksK]
    // gK: [kBlockN, kHeadDim, nblocksK]
    // gV: [kBlockN, kHeadDim, nblocksK]
    // nBlocksK = 1
    // I don't understand why Tri tiling doesn't work, this one from 66Ring works, but I see this tiling is similar with doing gemm tiling
    // all in row-major
    Tensor gQ = local_tile(
        mQ,
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_coord(m_block, _) // take the m_block-th rowblock
    );
    Tensor gK = local_tile(
        mK,
        Shape<Int<kBlockN>, Int<kHeadDim>>{}, 
        make_coord(n_block_max-1, _) // take the last colblocks, we do accumulate in reverse order, later explain
    );
    Tensor gV = local_tile(
        mV,
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_coord(n_block_max-1, _)
    );

    // tensors on shared memory sQ, sK, sV
    // shape
    // sQ: [kBlockM, kHeadDim]
    // sK: [kBlockN, kHeadDim]
    // sV: [kBlockN, kHeadDim]
    // all in row-major
    Tensor sQ = make_tensor(
        make_smem_ptr(reinterpret_cast<Element*>(smem_)),
        typename Kernel_traits::SmemLayoutQ{}
    );
    Tensor sK = make_tensor(
        sQ.data() + size(sQ),
        typename Kernel_traits::SmemLayoutKV{}
    );
    Tensor sV = make_tensor(
        sK.data() + size(sK),
        typename Kernel_traits::SmemLayoutKV{}
    );

    // tensor: sVt, sVtNoSwizzle
    // Why Vtransposed?
    // Because our MMA Atom does TN gemm, and later we have to do P @ V
    // In CuBLAS convention, we have A (m, k) and B (k, n)
    // TN setup: A transposed (k, m), B not transposed (k, n)
    // A (k, m) has k as the leading dimension --> cute intepretation: (m, k) : (k, 1)
    // B (k, n) has n as the leading dimension --> cute intepretation: (n, k) : (k, 1)
    // Look into P @ V, we have
    // P: (kBlockM, kHeadDim) : (kHeadDim, 1) --> already ok
    // V: (kHeadDim, kBlockN) : (kBlockN, 1) --> not ok
    // --> V needs to be transposed: V^T: (kBlockN, kHeadDim) : (kHeadDim, 1) --> ok
    
    // Why VtransposedNoSwizzle?
    // Because we have to do MMA thread partitioning on non-swizzled layout
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    // .get()?

    // G2S TiledCopy QKV
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));  // (QCPY, QCPY_M, QCPY_K)
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));  // (KCPY, KCPY_M, KCPY_K)
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));  // (VCPY, VCPY_M, VCPY_K)
    
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    // TiledMMA
    // do MMA partitioning for 2 gemms
    // tS means thread partitioning for gemm on S
    // tO means thread partitioning for gemm on O
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    // first gemm (line 8 of algo): Q @ K^T
    Tensor tSrQ = thr_mma.partition_fragment_A(gQ(_, _, 0));
    Tensor tSrK = thr_mma.partition_fragment_B(gK(_, _, 0));
    // second gemm (line 10 of algo): P @ V
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    
    // O accumulator
    // use tiled_mma to partition on O tile (kBlockM, kHeadDim)
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});   // (MMA=4, MMA_M, MMA_K)

    // S2R TiledCopy for 2 gemms
    // first gemm (line 8 of algo): Q @ K^T
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    // second gemm (line 10 of algo): P @ V
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    
    // --------- Inner Loop ---------- (line 6 - line 11 of algo)
    // we will do accumulate in reverse order: from n_block_max - 1 to 0 (or from T_c to 1 in the paper)
    // Tri noted the reason is that we only need 1 iterator n_block
    // and only last blocks need to be masked. Later we will see he separates computation between blocks that need and don't need mask
    
    // different with the algo in the paper, Tri deploys a pipeline for the inner loop
    // fetch Q rowblock to Smem (line 5 of algo)
    // prefetch last K colblock to Smem --> commit
    // inner loop
    //      wait<0>: all previous fetches are finished
    //      fetch corresponding V block to Smem --> commit
    //      gemm 1 (line 8 of algo): Q @ K^T (this doesn't need V block --> V loading and gemm 1 can be executed in parallel)
    //      wait<0>: previous fetch for V block is finished: V is ready
    //      fetch the next K colblock to Smem --> commit (the latest Kblock is now not needed)
    //      softmax_rescale_o (line 9 of algo)
    //      gemm 2 (line 10 of algo): P @ V

    // fetch Q rowblock to Smem

    cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);

    // prefetch the last K colblock
    cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    cute::cp_async_fence(); // commit

    // mask
    Mask mask;
    // softmax
    flash::Softmax<2 * size<1>(acc_o)> softmax;

    // handle blocks that need mask first
    constexpr int n_masking_steps = cute::ceil_div(kBlockM, kBlockN);

    clear(acc_o);
    // inner loop
    int n_block = n_block_max - 1;
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; masking_step++, n_block--)
    {
        // acc_s: this is for storing S and later P
        // use tiled_mma to partition on S tile (kBlockM, kBlockN)
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});   // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);

        // wait for previous fetches to finish
        flash::cp_async_wait<0>(); 
        __syncthreads();

        // fetch corresponding V block to Smem (advance gV)
        gV = local_tile(mV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(n_block, _));
        tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
        cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        cp_async_fence(); // commit

        // gemm 1: acc_s = Q @ K^T
        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        // mask acc_s
        mask.template apply_mask(
            acc_s,
            n_block * kBlockN,
            m_block * kBlockM,
            16 * kNWarps,
            8
        );

        // wait for previous V fetch to finish
        flash::cp_async_wait<0>(); 
        __syncthreads();

        // prefetch the next K colblock
        if (n_block > n_block_min)
        {
            gK = local_tile(mK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(n_block-1, _));
            tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
            cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
            cp_async_fence(); // commit
        }

        // softmax rescale
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2);

        // gemm 2: P @ V, update O
        Tensor rP = flash::convert_type<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) 
        // to ((4, 2), MMA_M, MMA_N/2) if using m16n8k16
        // or (4, MMA_M, MMA_N) if using m16n8k8
        // Why do we need to reshape?
        // Because other thread partitioning like tOrVt are done by tiled_mma
        // However rP is the output of gemm 1 and of shape (MMA=4, MMA_M, MMA_N) which is not compatible with tiled_mma
        // see the A layout of m16n8k16 in notebook/m16n8k16.cu
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // handle blocks that don't need mask
    for (; n_block >= n_block_min; n_block--)
    {
        // acc_s: this is for storing S and later P
        // use tiled_mma to partition on S tile (kBlockM, kBlockN)
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});   // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);

        // wait for previous fetches to finish
        flash::cp_async_wait<0>(); 
        __syncthreads();

        // fetch corresponding V block to Smem (advance gV)
        gV = local_tile(mV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(n_block, _));
        tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
        cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        cp_async_fence(); // commit

        // gemm 1: acc_s = Q @ K^T
        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        // wait for previous V fetch to finish
        flash::cp_async_wait<0>(); 
        __syncthreads();

        // prefetch the next K colblock
        if (n_block > n_block_min)
        {
            gK = local_tile(mK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(n_block-1, _));
            tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
            cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
            cp_async_fence(); // commit
        }
        
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2);

        // gemm 2: P @ V, update O
        Tensor rP = flash::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // epilouge
    softmax.template normalize_softmax_lse(acc_o); // I don't include LSE computation here but still call it because we need to do reduce over threads per row
    Tensor rO = flash::convert_type<Element>(acc_o);  // (MMA=4, MMA_M, MMA_N)

    // copy O from thread register to shared memory
    // sO
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    auto taccOrO = smem_thr_copy_O.retile_S(rO);
    auto taccOsO = smem_thr_copy_O.partition_D(sO);

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    // copy O from shared memory to global memory
    // mO
    // shape (seqlen_q, d)
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo.q_offset(params.o_batch_stride, params.o_head_stride, bidb, bidh)),
        make_shape(params.seqlen_q, params.d),
        make_stride(Int<kHeadDim>{}, _1{})
    );

    // gO
    Tensor gO = local_tile(
        mO,
        make_shape(kBlockM ,kHeadDim),
        make_coord(m_block, _)
    );

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    auto tOsO = gmem_thr_copy_O.partition_S(sO);
    auto tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));

    __syncthreads();

    cute::copy(gmem_tiled_copy_O, tOsO, tOgO);
}

template<typename Kernel_traits, bool Is_causal, typename Params> // Params is deduced from params
inline __device__ void compute_attn(const Params params)
{   
    // rowblock index
    const int m_block = blockIdx.x;
    // batch index
    const int bidb = blockIdx.y;
    // head index
    const int bidh = blockIdx.z;

    flash::compute_attn_1rowblock<Kernel_traits, Is_causal>(params, bidb, bidh, m_block);
}

} // end namspace flash