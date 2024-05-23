# pragma once

void self_attention_cuda(float *Q, float *K, float *V, float *O,
    int batch_size, int num_heads, int seq_len, int head_dim);