#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <utils.cuh>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Please select a kernel (range 0 - 11, here 0 is for NVIDIA cuBLAS).\n");
        exit(EXIT_FAILURE);
    }

    // cuda kernel num
    int kernel_num = atoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 11) {
        printf("Please enter a valid kernel number (0-11).\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Select kernel %d.\n", kernel_num);
    };

    // 핸들을 선언/생성합니다. cublasCreate는 cublasStatus_t 값을 반환하며(0이면 성공) 생성 여부를 판단할 수 있습니다.
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        printf("Create cublas handle error.\n");
        exit(EXIT_FAILURE);
    };

    // cudaEvent로 GPU 스트림 시간을 측정합니다. cudaEvent는 대상 스트림에 이벤트 작업을 게시하는 방식입니다.
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // matrix size
    int size_len = 24;
    int SIZE[size_len];
    for (int i = 0; i < size_len; i++)
        SIZE[i] = 256 * (i + 1);

    int m, n, k, max_size;
    max_size = SIZE[size_len - 1];
    printf("max_size=%d\n", max_size);

    float alpha = 1.0, beta = 0.; //two arbitary input parameters，C=α*AB+β*C

    float *A = NULL, *B = NULL, *C = NULL, *C_ref = NULL;     //host matrices
    float *dA = NULL, *dB = NULL, *dC = NULL, *dC_ref = NULL; //device matrices

    A = (float *) malloc(sizeof(float) * max_size * max_size);
    B = (float *) malloc(sizeof(float) * max_size * max_size);
    C = (float *) malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *) malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);
    copy_matrix(C, C_ref, max_size * max_size);

    cudaCheck(cudaMalloc((void **) &dA, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dC_ref, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C_ref, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    int repeat_times = 10;
    for (int i = 0; i < size_len; i++) {
        m = n = k = SIZE[i];

        printf("m=n=k=%d\n", m);
        // 계산 정확성을 검증하고, 콜드 스타트 오차를 줄이기 위해 계측 전 1회 선실행합니다.
        if (kernel_num != 0) {
            test_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);      // cuBLAS
            test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle); // user define
            cudaDeviceSynchronize();
            cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            if (!verify_matrix(C_ref, C, m * n)) {
                printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
                exit(EXIT_FAILURE);
            }
        }
        cudaDeviceSynchronize();

        cudaEventRecord(beg);
        for (int j = 0; j < repeat_times; j++) {
            test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.; // 초 단위로 변환

        printf("Average elasped time: (%f) second, performance: (%f) GFLOPS. size: (%d).\n",
               elapsed_time / repeat_times, 2. * 1e-9 * repeat_times * m * n * k / elapsed_time, m);
        fflush(stdout);
        copy_matrix(C_ref, C, m * n); //sync C with cuBLAS to prepare for the next run
    }

    // CPU/GPU 메모리 해제
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);

    return 0;
};
