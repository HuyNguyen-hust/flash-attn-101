#include "flash_fwd_launch_template.h"

// Explicitly instantiate for fp16, hdim 64
template<>
void run_mha_fwd_<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream)
{
    run_mha_fwd_hdim64<cutlass::half_t, true>(params, stream);
}