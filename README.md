# Flash Attention 101

## Overview

This repository provides a simplified implementation of the attention mechanism for self-learning purposes. It includes:

| Implementation | File Location |
|----------------|---------------|
| Naive Attention | `csrc/naive_attention.cu` |
| CUDA core Flash Attention 1 (FA1) | `csrc/flash_attn_1.cu` |
| CUDA core Flash Attention 2 (FA2) | `csrc/flash_attn_2.cu` |
| Tensor core FA2 using CUTLASS CuTe | `csrc/flash_attn/` |

## Build and Run
Use the provided CMakeLists.txt to build and run the CUDA programs:

```bash
git submodule init
git submodule update
cmake -B build
cmake --build build
./build/csrc/profile-attention
```

## Benchmark (On A100)
```
Device name: NVIDIA A100 80GB PCIe MIG 1g.10gb
Global memory size: 9 GB
Peak memory bandwidth: 241.92 GB/s

batch size = 8
sequence length = 256
number of heads = 16
dimension = 64
-------------------------------------------------
implementation: cuda core flash attention 01
all-close check passed
naive attention latency = 53.7733 ms
latency = 28.1272 ms
speedup = 191.179%

-------------------------------------------------
implementation: cuda core flash attention 02
all-close check passed
naive attention latency = 53.7846 ms
latency = 25.7505 ms
speedup = 208.868%

-------------------------------------------------
implementation: cute flash attention 02
all-close check passed
naive attention latency = 53.7733 ms
latency = 0.16384 ms
speedup = 32820.6%
```

## Usage
Customize the implementation by modifying the config in csrc/profile.cu. Note that this implementation supports head dimensions of 32 and 64 only.

## Implementation Details
The implementation in this repository follows the ideas and approaches presented in the following works:

- [https://github.com/Dao-AILab/flash-attention]
- [https://github.com/66RING/tiny-flash-attention]
- [https://github.com/tspeterkim/flash-attention-minimal]
- [https://github.com/leloykun/flash-hyperbolic-attention-minimal]
