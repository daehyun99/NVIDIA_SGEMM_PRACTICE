#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template<const int BM,
        const int BN,
        const int BK,
        const int TM,
        const int TN>
__global__ void mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread; // 스레드 1개가 블록 내 TM*TN개 원소 계산 담당

    // 현재 스레드가 담당하는 thread tile 좌상단 원소의 블록 내 위치
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];


    const int ldg_a_num = BK * BM / thread_num / 4; // 스레드마다 float 4개를 옮기며 As 로딩 완료까지 총 ldg_a_num 라운드 수행
    const int ldg_b_num = BK * BN / thread_num / 4; // 스레드마다 float 4개를 옮기며 Bs 로딩 완료까지 총 ldg_b_num 라운드 수행

    int a_tile_row = threadIdx.x / (BK / 4); // 행당 4바이트 단위를 메모리 블록으로 보고, 현재 스레드는 a_tile_row행의 a_tile_col 블록을 옮깁니다
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num; // 총 BM행을 ldg_a_num 라운드로 나누어 라운드당 a_tile_stride행씩 옮깁니다

    int b_tile_row = threadIdx.x / (BN / 4); // 행당 4바이트 단위를 메모리 블록으로 보고, 현재 스레드는 b_tile_row행의 b_tile_col 블록을 옮깁니다
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num; // 총 BK행을 ldg_b_num 라운드로 나누어 라운드당 b_tile_stride행씩 옮깁니다

    float accum[TM][TN] = {0.}; // 스레드당 TM*TN개 원소 계산을 위해 누산값 저장 레지스터 TM*TN개(및 추가 캐시 레지스터)가 필요합니다.

    // ldg_a_num 계산에 쓰는 모든 파라미터는 const여야 배열 크기 선언에 사용할 수 있습니다
    float ldg_a_reg[4 * ldg_a_num] = {0.}; // 각 스레드는 ldg_a_num 라운드 동안 ldg_a_num개의 float4를 레지스터에 저장해 As 전치에 사용

    float a_frag[TM];  // As 공유 메모리 캐시
    float b_frag[TN];  // Bs 공유 메모리 캐시

    // 현재 블록으로 포인터 이동
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

#pragma unroll
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;  // ldg_index번째 라운드
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            // As를 전치 저장합니다. ldg_a_reg는 중간 캐시이며 읽기 시 FLOAT4 로드를 가능하게 합니다
            As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
            As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
            As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // 전치 불필요
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++) {
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]); // 현재 thread tile 위치로 오프셋
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]); // 현재 thread tile 위치로 오프셋
            }
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            //float4 atmp = FETCH_FLOAT4(accum[m][n]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}