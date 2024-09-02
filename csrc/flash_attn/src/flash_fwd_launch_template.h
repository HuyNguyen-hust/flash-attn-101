#pragma once

#include "cuda_utils.hpp"

#include "flash.h"
#include "flash_fwd_kernel.h"

#include "utils.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// macro to clean up kernel definition
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template <typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_kwd_kernel, bool Is_causal)
{
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn<Kernel_traits, Is_causal>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// This is equivalent to
// template <typename Kernel_traits, bool Is_causal>
// global void flash_kwd_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)
// {
//     #if defined(ARCH_SUPPORTS_FLASH)
//         flash::compute_attn<Kernel_traits, Is_causal>(params);
//     #else
//         FLASH_UNSUPPORTED_ARCH
//     #endif
// }

template<
    typename Kernel_traits,
    bool Is_causal
>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream)
{
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    dim3 block(Kernel_traits::kNThreads);

    auto kernel = &flash_kwd_kernel<Kernel_traits, Is_causal>;
    if (smem_size >= 48 * 1024) // 48KB
    {
        CHECK_CUDA_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
    CHECK_LAST_CUDA_ERROR();
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream)
{
    constexpr static int Headdim = 32;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream)
{
    constexpr static int Headdim = 64;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, T>, Is_causal>(params, stream);
}