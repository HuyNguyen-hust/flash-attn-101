#include "flash_fwd_launch_template.h"

// Explicitly instantiate for fp16, hdim 32
template<>
void run_mha_fwd_<cutlass::half_t, 32, true>(Flash_fwd_params &params, cudaStream_t stream)
{
    run_mha_fwd_hdim32<cutlass::half_t, true>(params, stream);
}