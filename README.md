## Overview
This repository is made for self-taught purpose, providing a simplified implementation of the attention mechanism, covering the naive implementation, flash attention 1, and flash attention 2.

## Build and Run
This repository includes a CMakeLists.txt to automate the building and running of the CUDA programs. Simply run the following commands:

```!
cmake -B build
cmake --build build
./build/csrc/profile-attention
```

## Benchmark (On V100)
```
Device name: Tesla V100-SXM2-32GB
Global memory size: 31 GB
Peak memory bandwidth: 898.048 GB/s

batch size = 8
sequence length = 1024
number of heads = 16
dimension = 64
-------------------------------------------------
attention: attention 01
all-close check passed
naive attention latency = 751.191 ms
latency = 121.926 ms
speedup = 616.106%

-------------------------------------------------
attention: attention 02
all-close check passed
naive attention latency = 755.826 ms
latency = 79.3006 ms
speedup = 953.115%
```

## Usage
You can modify the config in csrc/profile.cu to customize the implementation according to your needs.

## Implementation Details
The implementation in this repository follows the ideas and approaches presented in the following works:

- [https://github.com/66RING/tiny-flash-attention]
- [https://github.com/tspeterkim/flash-attention-minimal]
- [https://github.com/leloykun/flash-hyperbolic-attention-minimal]

## Key Differences
Unlike above implementations this repository implements parallel computation over the sequence length (in flash attention 2), as suggested in the paper.
