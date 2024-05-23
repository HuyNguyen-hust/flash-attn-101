#pragma once

void flash_attn_2(float *Q, float *K, float *V, float *O,
    int batch_size, int num_heads, int seq_len, int head_dim);