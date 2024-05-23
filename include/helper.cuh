#pragma once

const int batch_size = 8;
const int num_heads = 12;
const int seq_len = 1024;
const int head_dim = 64;
const int Bc = 16;
const int Br = 16;

void print_host_tensor(float* matrix, int m, int n);
void print_device_tensor(float *matrix, int m, int n);
bool all_close(float *A, float *B, int batch_size, int num_heads, int seq_len, int head_dim);
