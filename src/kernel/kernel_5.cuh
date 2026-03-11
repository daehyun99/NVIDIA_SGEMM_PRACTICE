#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

template<const int BM,
        const int BN,
        const int BK,
        const int TM,
        const int TN>
__global__ void mysgemm_v5(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread; // 스레드 1개가 블록 내 TM*TN개 원소 계산 담당

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 현재 블록으로 포인터 이동
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    현재 스레드는 전역 메모리의 (a_tile_row, a_tile_col) 원소를 공유 메모리의 같은 좌표로 옮깁니다
    a_tile_stride는 블록 스레드가 공유 메모리로 옮길 수 있는 행 간격을 의미합니다.

    BM=64, BK=8, thread_num=512이면 a_tile_stride=64(=BM)로 각 스레드가 1회만 옮기면 됩니다.
    BM=128, BK=8, thread_num=512이면 a_tile_stride=64로 각 스레드가 2회 옮기면 됩니다.
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0.}; // 스레드당 TM*TN개 원소 계산을 위해 누산값 저장 레지스터 TM*TN개(및 추가 캐시 레지스터)가 필요합니다.
    float a_frag[TM] = {0.};
    float b_frag[TN] = {0.};

    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                a_frag[j] = As[(ty + j) * BK + i];
            }
            #pragma unroll
            for (int l = 0; l < TN; l++) {
                b_frag[l] = Bs[tx + l + i * BN];
            }
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                #pragma unroll
                for (int l = 0; l < TN; l++)
                    tmp[j][l] += a_frag[j] * b_frag[l];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
    }
}