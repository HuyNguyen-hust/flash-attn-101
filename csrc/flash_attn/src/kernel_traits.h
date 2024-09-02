#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

template<
    int kHeaddim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    typename elem_type=cutlass::half_t
>
struct Flash_kernel_traits
{   
    // Goal: precisions, G2S (gmem2smem) copy async flag, MMA Atom, and S2R (smem2reg) Copy Atom

    // precisions and G2S copy async
    // Since 800, cuda supports async copy from gmem to smem
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = uint32_t;

    // MMA Atom
    // since 800, cuda supports bf16
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;    
#endif

    // S2R Copy Atom
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif    
};

template<
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    typename elem_type=cutlass::half_t,
    typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>
>
struct Flash_fwd_kernel_traits
{
    // precisions 
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    
    // G2S async flag
    static constexpr bool Has_cp_async = Base::has_cp_async;
    
    // S2R Copy Atom
    // No tiling, because we will do it later with information from TiledMMA
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // the number of threads
    static constexpr int kNWarps = kNWarps_;  
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    // TiledMMA
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>, _1, _1>>,    // Threads repeated by NWarps x 1 x 1
        Tile<Int<16 * kNWarps>, _16, _16>         // No Values repeated
    >;

    // SmemLayout for Q, KV, Vtransposed, O, and Oaccum
    
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
    // Because we need to do MMA thread partition on non-swizzled layout
    // Remember swizzle is for copy, and non-swizzle is for MMA
    
    using SmemLayoutAtomQ = decltype(
        composition(
            Swizzle<kSwizzle, 3, 3>{},
            Layout<
                Shape<_8, Int<kBlockKSmem>>,    // 8 x kBlockKSmem
                Stride<Int<kBlockKSmem>, _1>    // Q is in row-major
            >{}
        )
    );
    // Tri noted that it must be kBlockKSmem not kHeadDim
    // [https://github.com/66RING/tiny-flash-attention/issues/7]

    using SmemLayoutQ = decltype(
        tile_to_shape(
            SmemLayoutAtomQ{},
            Shape<Int<kBlockM>, Int<kHeadDim>>{}
        )
    );

    using SmemLayoutKV = decltype(
        tile_to_shape(
            SmemLayoutAtomQ{},
            Shape<Int<kBlockN>, Int<kHeadDim>>{}
        )
    );

    using SmemLayoutVtransposed = decltype(
        composition(
            SmemLayoutKV{},
            make_layout(
                Shape<Int<kHeadDim>, Int<kBlockN>>{},
                GenRowMajor{}
            )
        )
    );
    // Given layout A = (row, col) : (col, 1)
    // --> layout A^T = (col, row) : (1, col)
    // Prove layout A^T = A o (col, row) : (row, 1)
    // Proof
    // A o (col, row) : (row, 1)
    // = (A o (col : row), A o (col : 1)) (*)
    // We have
    // A o (col : row) = ((row, col): (col, 1)) o (col, row) = col : 1
    // A o (col : 1) = ((row, col): (row, 1)) o (col, 1) = row : col
    // (*) = ((row : 1), (col : row)) = (col, row) : (1, col) (Q.E.D.)
    
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));
    // What does SmemLayoutVTransposedNoSwizzle look like?
    // It is the SmemLayoutVTransposed if you start with non-swizzle SmemLayoutAtomQ
    // run notebook/get_nonswizzle_portion.cu for better understanding

    // // Similar to 66Ring code
    // using SmemLayoutVtAtom = decltype(
    //     composition(Swizzle<kSwizzle, 3, 3>{},
    //                 Layout<Shape<Int<kBlockKSmem>, Int<kBlockN>>,
    //                        Stride<_1, Int<kBlockKSmem>>>{}));

    // using SmemLayoutVtransposed = decltype(tile_to_shape(
    //                                     SmemLayoutVtAtom{},
    //                                     Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    // using SmemLayoutVtransposedNoSwizzle = Layout<Shape<Int<kHeadDim>, Int<kBlockN>>,
    //                                             Stride<_1, Int<kHeadDim>>>;

    using SmemLayoutAtomO = decltype(
        composition(
            Swizzle<kSwizzle, 3, 3>{},
            Layout<
                Shape<_8, Int<kBlockKSmem>>,    // 8 x kBlockKGmem
                Stride<Int<kBlockKSmem>, _1>    // O is in row-major
            >{}
        )
    );
    using SmemLayoutO = decltype(
        tile_to_shape(
            SmemLayoutAtomO{},
            Shape<Int<kBlockM>, Int<kHeadDim>>{}
        )
    );

    // SmemCopyAtom for O
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    // SmemSize
    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * sizeof(Element) * 2;
    static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

    // G2S TiledCopy
    // TiledCopy needs copy atom, thread layout, and value layout
    
    // GmemTiledCopyQKV
    // copy struct --> copy atom
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    // each copy loads 128 bits = 8 halfs --> value layout must be contiguous and have size of multiples of 8 to do vectorized loads
    // --> choose value layout: <_1, _8>, it means 1 thread will do 1 load(8 halfs)
    // If it is <_1, _16>, it means 1 thread will do 2 loads
    
    // thread layout: size of GmemLayoutAtom must be kNThreads
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad; // because 1 thread ~ 1 load
    static_assert(kBlockKSmem % kGmemElemsPerLoad == 0, "kBlockKSmem must be a multiple of kGmemElemsPerLoad");
    using GmemLayoutAtom = decltype(
        Layout<
            Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>
        >{}
    );

    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(
            Copy_Atom<Gmem_copy_struct, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{} // row-major
        )
    );

    // GmemTiledCopyO
    // copy atom: DefaultCopy
    // thread layout: GmemLayoutAtom
    // value layout: <_1, _8>
    // 1 thread load 8 halfs with 8 loads because DefaultCopy loads 1 element at a time
    using GmemTiledCopyO = decltype(
        make_tiled_copy(
            Copy_Atom<DefaultCopy, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{} // row-major
        )
    );

    // GmemTiledCopyOaccum
    // copy atom: DefaultCopy
    // thread layout: GmemLayoutAtomOaccum
    // value layout: <_1, _4>
    // 1 thread load 4 halfs with 4 loads because DefaultCopy loads 1 element at a time
    // using GmemLayoutAtomOaccum = std::conditional_t<
    //     kBlockKSmem == 32,
    //     Layout<Shape <_16, _8>,  // Thread layout, 8 threads per row
    //            Stride< _8, _1>>,
    //     Layout<Shape <_8, _16>,  // Thread layout, 16 threads per row
    //            Stride< _16, _1>>
    // >;

    // using GmemTiledCopyOaccum = decltype(
    //     make_tiled_copy(
    //         Copy_Atom<DefaultCopy, ElementAccum>{},
    //         GmemLayoutAtomOaccum{},
    //         Layout<Shape<_1, _4>>{} // row-major
    //     )
    // );
};