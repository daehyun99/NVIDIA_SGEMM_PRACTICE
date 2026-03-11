#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*
=====================================
CUDA 작업
=====================================
*/
void cudaCheck(cudaError_t error, const char *file, int line); //CUDA 오류 검사
void CudaDeviceInfo();                                         // CUDA 정보 출력

/*
=====================================
행렬 작업
=====================================
*/
void randomize_matrix(float *mat, int N);            // 행렬 무작위 초기화
void copy_matrix(float *src, float *dest, int N);    // 행렬 복사
void print_matrix(const float *A, int M, int N);     // 행렬 출력
bool verify_matrix(float *mat1, float *mat2, int N); // 행렬 검증

/*
=====================================
시간 측정 작업
=====================================
*/
float get_current_sec();                        // 현재 시각 획득
float cpu_elapsed_time(float &beg, float &end); // 경과 시간 계산

/*
=====================================
커널 작업
=====================================
*/
//지정한 커널 함수를 호출해 행렬 곱 계산
void test_kernel(int kernel_num, int m, int n, int k, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle);