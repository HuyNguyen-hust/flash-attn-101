#include <stdio.h>
#include <cmath>

void print_host_tensor(float* matrix, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

void print_device_tensor(float *matrix, int m, int n) {
    float* host_matrix = (float*)(malloc(sizeof(float) * m * n));
    cudaMemcpy(host_matrix, matrix, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", host_matrix[i * n + j]);
        }
        printf("\n");
    }
}

bool all_close(float *A, float *B, int batch_size, int num_heads, int seq_len, int head_dim) {
    for (int i = 0; i < batch_size * num_heads * seq_len * head_dim; i++) {
        if (fabs(A[i] - B[i]) > 1e-5) {
            printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, B[i]);
            return false;
        }
    }
    return true;
}