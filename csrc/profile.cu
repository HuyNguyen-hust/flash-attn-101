#include <iostream>
#include <vector>
#include <cuda_fp16.h>
#include <functional>

#include "cuda_attn.hpp"
#include "profile_utils.cuh"

int main()
{   
    print_device_info();
    unsigned int batch_size = 8U;
    unsigned int num_heads = 16U;
    unsigned int seq_len = 256U;
    unsigned int head_dim = 64U; // just implement for 32 and 64

    unsigned int num_warmups = 1U;
    unsigned int num_repeats = 1U;

    __half const abs_tol{__float2half(5.0e-2f)};
    double const rel_tol{1.0e-1f};

    // print attention settings
    std::cout << "batch size = " << batch_size << std::endl;
    std::cout << "sequence length = " << seq_len << std::endl;
    std::cout << "number of heads = " << num_heads << std::endl;
    std::cout << "dimension = " << head_dim << std::endl;

    const std::vector <
        std::pair<
            std::string,
            std::function<
                void(
                    const __half*,
                    const __half*,
                    const __half*,
                    __half*,
                    unsigned int, unsigned int, unsigned int, unsigned int,
                    cudaStream_t stream
                )
            >
        >
    > attention_launchers = {
        {"cuda core flash attention 01", launch_flash_attention_01<__half>},
        {"cuda core flash attention 02", launch_flash_attention_02<__half>},
        {"cute flash attention 02", mha_fwd<__half>},
    };

    for (const auto& [name, attention_launcher] : attention_launchers) {
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "implementation: " << name << std::endl;
        std::pair<float, float> results = profile_attention<__half>(
            batch_size, num_heads, seq_len, head_dim,
            attention_launcher,
            num_warmups, num_repeats,
            abs_tol, rel_tol
        );
        std::cout << std::endl;
    }

    return 0;
}