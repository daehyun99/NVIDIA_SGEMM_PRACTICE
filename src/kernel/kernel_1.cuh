#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

__global__ __launch_bounds__(1024) void
mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int gx = blockIdx.x * blockDim.x + threadIdx.x; // 전역 x
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // 전역 y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[gy * K + i] * B[i * N + gx]; // 전역 메모리 2회 접근 + FMA(곱-누산) 1회
    }
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}