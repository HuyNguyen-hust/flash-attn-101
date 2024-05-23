## Overview
This repository is made for self-taught purpose, providing a simplified implementation of the attention mechanism, covering the naive implementation, flash attention 1, and flash attention 2.

## Key Differences
Unlike above implementations this repository implements parallel computation over the sequence length (in flash attention 2), as suggested in the paper.

## Build and Run
This repository includes a Makefile to automate the building and running of the CUDA programs. Simply run the following command:

```!
make
```

## Benchmark (On V100)
Config:
- batch_size = 8
- num_heads = 12
- seq_len = 1024
- head_dim = 64
- Bc = 16
- Br = 16

```
----------- naive self attention -----------
Time for kernel execution: 4.229 ms 
----------- flash attention 1 ----------- 
Time for kernel execution: 1.225 ms 
----------- flash attention 2 ----------- 
Time for kernel execution: 0.601 ms 
----------- sanity check ----------- 
all close: 1
```

## Usage
You can modify the config in include/helper.cuh to customize the implementation according to your needs.

## Implementation Details
The implementation in this repository follows the ideas and approaches presented in the following works:

- [https://github.com/66RING/tiny-flash-attention]
- [https://github.com/tspeterkim/flash-attention-minimal]
- [https://github.com/leloykun/flash-hyperbolic-attention-minimal]
