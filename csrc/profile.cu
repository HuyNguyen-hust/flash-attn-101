#include <iostream>
#include <vector>
#include <functional>

#include "cuda_attn.hpp"
#include "profile_utils.cuh"

int main()
{   
    print_device_info();
    unsigned int batch_size = 8U;
    unsigned int num_heads = 16U;
    unsigned int seq_len = 1024U;
    unsigned int head_dim = 64U;

    unsigned int num_warmups = 1U;
    unsigned int num_repeats = 1U;

    float abs_tol = 1.0e-3f;
    double rel_tol = 0.0e-4f;

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
                    const float*,
                    const float*,
                    const float*,
                    float*,
                    unsigned int, unsigned int, unsigned int, unsigned int,
                    cudaStream_t stream
                )
            >
        >
    > attention_launchers = {
        {"attention 01", launch_flash_attention_01<float>},
        {"attention 02", launch_flash_attention_02<float>}
    };

    for (const auto& [name, attention_launcher] : attention_launchers) {
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "attention: " << name << std::endl;
        std::pair<float, float> results = profile_attention<float>(
            batch_size, num_heads, seq_len, head_dim,
            attention_launcher,
            num_warmups, num_repeats,
            abs_tol, rel_tol
        );
        std::cout << std::endl;
    }

    return 0;
}