![](images/head.png)

![](https://img.shields.io/badge/build-passing-brightgreen) ![](https://img.shields.io/badge/ubuntu-18.04-blue) ![](https://img.shields.io/badge/cuda-10.2-blue) ![](https://img.shields.io/badge/nvidia-RTX3090-blue) ![](https://img.shields.io/badge/cmake-3.21-blue)



# 개요

NVIDIA GPU를 대상으로, CUDA 프로그래밍을 사용하여 행렬 곱셈 연산 성능을 단계별로 최적화합니다:

| 커널 함수 | 설명                    | GFLOPS   | 사용자 정의 커널 함수/CUBLAS (%) |
| -------- | ----------------------- | -------- | ------------------------ |
| CUBLAS   | 공식 라이브러리 함수           | 14448.69 | 기준                      |
| kernel_1 | 단순(Naive) 구현            | 2262.168 | 15.65657                 |
| kernel_2 | 공유 메모리 캐시             | 4216.536 | 29.18283                 |
| kernel_3 | 1차원 Thread Tile 병렬 최적화 | 7809.629 | 54.05078                 |
| kernel_4 | 2차원 Thread Tile 병렬 최적화 | 12251.3  | 84.79179                 |
| kernel_5 | 레지스터 캐시                | 12177.95 | 84.28412                 |
| kernel_6 | FLOAT4 벡터 메모리 접근       | 13161.49 | 91.09125                 |
| kernel_7 | 이중 버퍼 프리패치            | 13634.98 | 94.36832                 |

> NVIDIA GeForce RTX 3090, 행렬 크기 5120

# 환경 설정

- 컴파일은 Ubuntu 18.04.5 LTS 환경에서 `gcc 7.5.0`을 사용합니다.
- NVIDIA CUDA version: `CUDA 10.2`；

```
# 디렉토리 구조
NVIDIA_SGEMM_PRACTICE                                   # 루트 디렉토리
├── images                                          # 이미지 결과
│     ├── describe_kernel_1.png

│     ├── describe_kernel_x.png
│     └── kernel_x_vs_y.png
├── test                                            # 테스트 결과
│     ├── test_kernel_0.txt
│     ├── test_kernel_1.txt
│     └── test_kernel_x.txt
└── src                                             # 소스 파일
│    ├── kernel
│    │  ├── kernel_1.cuh                            # 선언 및 정의
│    │  ├── kernel_2.cuh
│    │  └── kernel_x.cuh
│    ├── kernel.cuh
│    ├── utils.cuh                                  # 보조 함수
│    └── utils.cu
├── plot.py                                         # test 결과를 바탕으로 그래프 그리기
├── run.sh                                          # 컴파일된 실행 파일 실행
├── sgemm.cu                                        # 메인 프로그램
└── CMakeLists.txt                                  # 컴파일 관련

```

# 실행 방법
1. NVCC 컴파일 매개변수 설정
> CMakeLists.txt에서 `set(CUDA_NVCC_FLAGS -arch=compute_70;-code=compute_70)`를 수정합니다.
2. 행렬 계산 최대 크기 설정
> `sgemm.cu:16`에서 `size_len`을 수정합니다. 처음 실행 시에는 16으로 설정하는 것을 권장하며, 크기가 너무 크면 전원 과부하로 인해 호스트가 재부팅될 수 있습니다.
3. 컴파일
`cd build && cmake .. && make`
4. run.sh를 실행하여 각 커널 함수의 계산 효율을 통계 내고, 결과는 test 디렉토리에 저장합니다.
5. 계산 효율 꺾은선 그래프 그리기

> `python plot.py 0 1`은 CUBLAS와 kernel_1의 계산 효율 비교 그래프를 그린다는 의미입니다.

# 단계별 최적화

##  kernel 1 

**Naive 기본 버전 행렬 곱셈 구현**

각 논리 스레드를 행렬 C의 각 요소에 대응시켜, 각 스레드가 C의 요소 하나에 대한 계산을 담당하도록 합니다.

![](./images/describe_kernel_1.png)

```cpp
__global__ __launch_bounds__(1024) void
mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int gx = blockIdx.x * blockDim.x + threadIdx.x; // 전역 x
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // 전역 y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[gy * K + i] * B[i * N + gx]; // 두 번의 전역 메모리 접근과 한 번의 FMA(곱셈-누산)
    }
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}

```

최적화되지 않은 행렬 곱셈의 성능은 CUBLAS의 1/10에도 미치지 못하며, 구체적인 분석은 다음과 같습니다.

* 계산 대 메모리 접근 비율(Compute-to-Memory-Access Ratio): 각 반복마다 한 번의 FMA(곱셈-누산)와 두 번의 전역 메모리 읽기가 필요하므로, 계산 대 메모리 접근 비율은 1/2입니다.
* 메모리 접근량: 전역 메모리에 접근할 때, 행렬 C의 각 요소를 계산하려면 `2K`개의 단정밀도 부동소수점을 읽어야 하며, 전체 계산을 완료하려면 `2*K*M*N`번 접근해야 합니다.

전역 메모리는 접근 지연 시간이 길고(수백 사이클), 동일한 위치의 요소가 반복적으로 읽힙니다(C의 같은 행 요소들을 계산할 때 A의 같은 행 요소들을 공유하고, C의 같은 열 요소들을 계산할 때 B의 같은 열 요소들을 공유함). 다른 한편으로, 낮은 계산 대 메모리 접근 비율로는 메모리 접근 지연을 효과적으로 숨길 수 없습니다. 따라서 메모리 접근 지연 시간과 계산 대 메모리 접근 비율이 kernel 1의 효율을 떨어뜨리는 원인입니다.

## kernel 2

**공유 메모리 캐시를 활용하여 전역 메모리 접근량 및 접근 지연 시간 감소**

메모리 접근 지연은 전역 메모리의 높은 지연 시간과 반복적인 접근에서 비롯됩니다. 공유 메모리는 온칩(on-chip) 메모리로서 접근 지연 시간이 짧으므로(수십 사이클), 공유 메모리를 캐시로 사용하면 접근 지연 시간을 줄일 수 있습니다.

> BM과 BN은 block tile의 높이와 너비를 나타내며, BK는 캐시할 전역 메모리의 보폭(stride)을 나타냅니다. 즉, 하나의 block을 계산하려면 K/BK번 캐시해야 합니다.

공유 메모리에 전역 메모리의 A tile과 B tile을 캐시하고, C block 내 모든 요소의 FMA 계산을 완료한 뒤, 캐시 영역을 계속 슬라이딩하며 block을 업데이트합니다.

```cpp
/*
dim3 blockDim(1024);
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
mysgemm_v2<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
*/

template<const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    // 공유 메모리 공간 할당
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 현재 block으로 이동
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {
        // A_tile과 B_tile 캐시
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // 모든 스레드의 캐시 완료 동기화
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA 계산 시 캐시 데이터를 읽어야 하므로, 새로운 라운드의 캐시 쓰기 전에 동기화하여 모든 스레드의 계산 완료를 보장
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}

```

* 메모리 접근량: 각 block은 전역 메모리에서 `(K/BK)*(BM*BK+BK*BN)`개의 단정밀도 부동소수점을 읽어와야 하며, 전체 C에는 `(M/BM)*(N/BN)`개의 block이 존재합니다. 따라서 C의 모든 요소 계산을 완료하려면 총 `(M/BM)*(N/BN)*(K/BK)*(BM*BK+BK*BN)`개의 단정밀도 부동소수점을 읽어와야 합니다.

kernel 1은 전역 메모리의 접근 지연과 반복적인 접근으로 인해 제한을 받았습니다. 최적화 전 전역 메모리 접근량은 `2*K*M*N`이었으나, 공유 메모리 캐시 최적화 후에는 원래의 `1/2*(1/BN)*(1/BM)`로 감소하여, `BN=BM=32`일 때 접근량은 1/32로 줄어듭니다. 한편 공유 메모리의 접근 지연은 전역 메모리보다 훨씬 낮기 때문에 계산 효율이 어느 정도 향상되었습니다.

## kernel 3

**1차원 thread tile을 활용한 최적화**

block 크기(BM, BN)를 늘림으로써 전역 메모리 접근량을 더욱 줄일 수 있다는 점을 알았으므로, BM과 BN을 32에서 64로 늘립니다.

> **block 크기를 무한정 늘려서 전역 메모리 접근을 줄일 수 있을까요?**
> 불가능합니다. 첫째, block 분할 행렬 크기가 너무 커지면 block 수가 줄어들어 대량의 SM(Streaming Multiprocessor)이 유휴 상태로 낭비될 수 있습니다. 둘째, BN과 BM이 증가하면 더 많은 공유 메모리를 할당해야 하는데, 단일 스레드 내에서 공유 메모리 점유율이 높아질수록 활성 스레드 워프(Warp) 수가 줄어들어 명령어 지연 시간을 숨기는 데 불리해집니다.

따라서 BM과 BN 값을 늘리는 동시에 공유 메모리 점유를 줄이기 위해 한편으로는 BK 값을 8로 낮춥니다.

> block 크기를 늘릴 때는 특히 공유 메모리 소모에 주의해야 하며, 자원 부족으로 커널 함수가 실행되지 않는 것을 방지하기 위해 공유 메모리 크기와 block 내 스레드 수를 제한해야 합니다.

다른 한편으로, 공유 메모리 캐시를 통해 전역 메모리 접근량과 FMA 곱셈-누산의 메모리 접근 지연은 줄였지만 계산 대 메모리 접근 비율은 개선되지 않았습니다. 매 반복 계산마다 두 개의 메모리 접근 명령어와 한 개의 계산 명령어가 필요합니다. 따라서 thread tile을 도입하여, 한 스레드가 block 내의 여러 요소 계산을 담당하도록 합니다. 여기서 TM과 TN은 각각 thread tile의 높이와 너비를 나타냅니다.

```cpp
/*
dim3 blockDim(512);
dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
mysgemm_v3<64, 64, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
*/


template<const int BM,
        const int BN,
        const int BK,
        const int TM>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_num = BM * BN / TM; // 하나의 스레드가 block 내에서 TM개의 요소 계산을 담당

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 현재 block으로 이동
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    현재 스레드는 전역 메모리의 a_tile_row행, a_tile_col열 요소를 공유 메모리의 a_tile_row행, a_tile_col열로 운반하는 역할을 합니다.
    a_tile_stride는 block 내의 스레드들이 a_tile_stride개의 행을 공유 메모리로 운반할 수 있음을 나타냅니다.

    만약 BM=64, BK=8, thread_num=512라면 a_tile_stride=64, a_tile_stride=BM이 되어 각 스레드가 한 번만 운반하면 필요한 요소의 운반을 완료할 수 있음을 의미합니다.
    만약 BM=128, BK=8, thread_num=512라면 a_tile_stride=64가 되어 각 스레드가 두 번 운반해야 필요한 요소의 운반을 완료할 수 있음을 의미합니다.
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM + 1] = {0.}; // 각 스레드가 TM개의 요소를 담당하므로 누적 값을 저장할 TM개의 레지스터가 필요하며, 추가로 캐시용 레지스터 1개가 필요합니다.
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
            tmp[TM] = Bs[tx + i * BN]; // 추가 레지스터 1개를 사용하여 공유 메모리에서 Bs[tx + i * BN]을 반복해서 읽는 것을 방지
            #pragma unroll  // 루프 언롤링(Loop unrolling)으로 명령어 병렬성 증가
            for (int j = 0; j < TM; j++) {
                tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < TM; j++) {
        C[(ty + j) * N + tx] = alpha * tmp[j] + beta * C[(ty + j) * N + tx];
    }
}

```

이 예제에서는 다음 두 가지 측면에서 최적화를 진행합니다:

* 전역 메모리 접근량: 초기 버전과 비교하여 `64*64` block 크기를 캐싱함으로써 메모리 접근량을 1/64로 줄였습니다.
* 계산 대 메모리 접근 비율: thread tile을 도입하여 단일 스레드가 여러 요소의 계산을 담당하게 함으로써 계산 대 메모리 접근 비율을 높였습니다. TM=8일 때 공유 메모리 As에 대한 8번의 접근 명령어와 Bs에 대한 1번의 접근 명령어를 실행할 때마다 8번의 계산 명령어를 실행할 수 있습니다. 이는 초기 버전의 계산 대 메모리 접근 비율 1:2와 비교하여 8:9로 향상된 것이며, 메모리 접근 지연을 효과적으로 숨길 수 있습니다.

이 예제의 두 가지 최적화를 통해 행렬 곱셈의 계산 효율이 거의 두 배 가까이 크게 향상되었습니다.

## kernel 4

**2차원 thread tile을 활용한 최적화**

thread tile을 2차원으로 설정하여 하나의 스레드가 작은 블록의 요소들을 계산하도록 담당하게 함으로써, block 크기를 더욱 늘리고 전역 메모리 접근 횟수를 줄입니다.

> thread tile 크기를 늘리면 동일하거나 더 적은 스레드 수로 더 큰 block 크기를 계산할 수 있습니다.

더 중요한 점은, 단일 스레드가 더 많은 C 요소 영역의 계산을 담당하게 되어 명령어 수준의 병렬성을 높일 수 있다는 것입니다.

> 왜 명령어 병렬성을 높일 수 있을까요?
> 단일 스레드가 처리하는 명령어 수가 많을수록 파이프라인 단계가 길어집니다. 단일 스레드 파이프라인은 여러 명령어를 병렬로 처리할 수 있으므로, 단일 명령어의 실행은 느려질지라도 단위 시간당 처리하는 명령어 수가 많아져 처리량(Throughput)이 향상되고 명령어 지연이 숨겨집니다. 명령어 수준 병렬성은 스레드 수준 병렬성보다 이점이 큽니다.

한 스레드가 8×8 영역 내의 요소 계산을 담당하도록 설정합니다. 즉, thread tile=8×8, TM=8, TN=8입니다.

```cpp
// BM=BN=128, BK=8, TM=TN=8, 공유 메모리 크기 128*8
dim3 blockDim(256);
dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
mysgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;  // 128*8/256=4, 모든 스레드가 4번씩 운반해야 하며, 이를 통해 전역 메모리의 128*8 크기 영역을 공유 메모리로 운반할 수 있음

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

// 각 스레드는 TM*TN개의 요소를 담당하므로, 누적 값을 저장할 TM*TN개의 레지스터가 필요함;
float tmp[TM][TN] = {0.}; 

// 단일 스레드가 TM, TN만큼 루프를 돌아 thread tile 내 요소의 곱셈-누산을 완료
for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
        tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
}

```

전역 메모리 접근량: 공유 메모 캐시를 도입하지 않은 버전에 비해 전역 메모리 접근량이 `1/2*(1/BM+1/BN)=1/128`로 감소하여 접근량이 크게 줄었습니다.

실제 테스트 결과, 1차원 thread tile에 비해 2차원 thread tile은 전역 메모리 접근량을 더욱 낮추고 계산 대 메모리 접근 비율을 높였기 때문에 행렬 곱셈 효율이 눈에 띄게 두 배로 향상되었습니다.

## kernel 5

**레지스터를 활용한 공유 메모리 캐싱**

아래 코드에서 알 수 있듯이, 단일 스레드가 thread tile 요소의 곱셈-누산을 계산할 때 공유 메모리에 반복적으로 접근하게 됩니다.

```cpp
for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
        tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];  // 내부 루프에서 As[(ty + j) * BK + i]에 TN번 반복 접근함
}

```

공유 메모리는 전역 메모리에 비해 메모리 접근 지연을 크게 줄일 수 있지만, 공유 메모리 지연 시간(수십 사이클)은 연산 지연 시간(수 사이클)에 비하면 여전히 큽니다. 따라서 공유 메모리의 반복적인 접근을 피하기 위해 레지스터를 사용하여 공유 메모리 As, Bs를 캐싱합니다.

```cpp
float a_frag[TM] = {0.};
float b_frag[TN] = {0.};

for (int i = 0; i < BK; i++) {
    for (int j = 0; j < TM; j++) {
        a_frag[j] = As[(ty + j) * BK + i];     // a_frag 레지스터 배열을 사용하여 thread tile에 필요한 As 공유 메모리 데이터를 캐싱;
    }
    for (int l = 0; l < TN; l++) {
        b_frag[l] = Bs[tx + l + i * BN];       // b_frag 레지스터 배열을 사용하여 thread tile에 필요한 Bs 공유 메모리 데이터를 캐싱;
    }
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            tmp[j][l] += a_frag[j] * b_frag[l];
    }
}

```

TM=TN=8일 때 레지스터 캐싱을 거치면 각 thread tile은 8개의 As 공유 메모리 접근 명령어와 8개의 Bs 공유 메모리 접근 명령어를 실행해야 하며, 8×8=64개의 계산 명령어를 수행할 수 있습니다. 계산 대 메모리 접근 비율은 초기 버전의 1/2에서 64:16으로 향상되어 메모리 접근 지연을 효과적으로 숨길 수 있습니다.

실제 테스트 결과, 레지스터 캐싱을 적용해도 실제 성능에 큰 변화는 없었습니다. 그 이유는 현재의 성능 병목이 공유 메모리의 반복적인 접근에 있지 않기 때문일 수 있습니다.

## kernel 6

**벡터 메모리 명령어 FLOAT4 최적화**

* 계산 명령어: GPU는 4차원 벡터를 기본 단위로 계산을 수행합니다. 4개의 부동소수점으로 구성된 float4 벡터는 GPU의 가장 기본적인 타입이며, GPU를 사용하여 두 개의 float4에 대해 벡터 계산을 하는 것은 두 개의 정수나 부동소수점을 계산하는 것과 마찬가지로 단 하나의 명령어만으로 완료됩니다.
* 메모리 명령어: 단일 명령어를 내려 개별 메모리 트랜잭션을 생성해 동일한 바이트 수를 가져오는 것과 비교할 때, 벡터 메모리 명령어를 통하면 필요한 메모리 트랜잭션 수가 더 적어져 메모리 컨트롤러의 경합이 줄어듭니다. 다른 한편으로, 벡터 로드를 사용하면 각 바이트당 더 적은 인덱스 계산이 필요합니다.

예를 들어 BM=128, BK=8, 스레드 수가 256일 때, 각 스레드가 매번 1개의 부동소수점을 가져온다면 한 스레드당 4번의 메모리 명령어를 소비해야만 전역 메모리를 공유 메모리로 운반할 수 있습니다. 반면 float4 벡터 메모리 명령어를 사용하면 각 스레드가 한 번에 4개의 부동소수점을 운반할 수 있어, 스레드당 단 한 번의 메모리 명령어 실행만으로 운반을 완료할 수 있습니다.

핵심 코드 예시는 다음과 같습니다:

```cpp
#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float ldg_a_reg[4 * ldg_a_num] = {0.}; // 각 스레드가 ldg_a_num 라운드 운반하며, 레지스터에 ldg_a_num개의 float4 요소를 캐싱하여 As 행렬을 전치(Transpose)하는 데 사용

//  공유 메모리에 전역 메모리 캐시
for (int i = 0; i < BM; i += a_tile_stride) {
    int ldg_index = i / a_tile_stride * 4;  // ldg_index번째 라운드
    FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
            FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
    // As 전치 저장. 이때 ldg_a_reg를 중간 캐시로 사용하며, 읽을 때 FLOAT4 단위로 읽을 수 있도록 하는 것이 목적
    As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
    As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
    As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
    As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
}

for (int i = 0; i < BK; i += b_tile_stride) {
    FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
        FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // 전치 불필요
}


// 레지스터를 활용한 공유 메모리 캐싱
// ty, tx는 현재 스레드에 대응하는 thread tile의 좌상단 요소가 block 내에서 차지하는 위치
#pragma unroll
for (int m = 0; m < TM; m += 4) {
    FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]); // 현재 thread tile로 오프셋 이동
}
#pragma unroll
for (int n = 0; n < TN; n += 4) {
    FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]); // 현재 thread tile로 오프셋 이동
}

```

전역 메모리는 공유 메모리에 직접 쓸 수 없으며 매개체인 레지스터가 필요합니다. 그중 As를 쓸 때는 전역 메모리 -> 레지스터 -> 공유 메모리로 이어지는 과정을 명시적으로 기술했습니다. Bs를 쓸 때 레지스터가 필요 없는 것이 아니라 컴파일러가 해당 코드를 숨겼을 뿐입니다. As 캐시에서 레지스터 사용을 명시한 목적은 As를 전치(Transpose)하기 위함입니다. 전치 전의 열(column)이 전치 후 행(row)으로 바뀌면 메모리가 연속적이게 되어 float4로 읽기 편해집니다.

실제 테스트 결과 전체 계산 효율이 증가했습니다.

## kernel 7

**데이터 프리패치(Data Prefetching)**

단일 버퍼(Single Buffer)는 단일 블록의 공유 메모리를 할당하여 전역 데이터를 캐시하고 단일 블록의 레지스터 메모리를 할당하여 공유 데이터를 캐시하는 것을 의미합니다. 단일 버퍼는 데이터 간의 의존성 때문에 읽기와 쓰기를 병렬로 수행할 수 없습니다. 예를 들어 단일 버퍼 시나리오에서 계산은 공유 메모리 데이터에 의존하므로, 계산 전에 전역 메모리가 공유 메모리에 완전히 저장되도록 보장하기 위해 한 번의 동기화가 필요합니다. 마찬가지로 계산이 공유 메모리 데이터에 의존하기 때문에 다음 라운드의 전역 메모리를 공유 메모리에 저장하기 전에도 한 번 동기화를 수행하여 이전 라운드의 계산이 완료되었음을 보장해야 합니다.

이중 버퍼(Double Buffer)는 두 배의 저장 공간을 할당하여 읽기와 쓰기를 분리합니다. 데이터를 계산하기 위해 하나의 저장 공간을 읽는 동시에, 다음 라운드에서 의존할 데이터를 다른 메모리 블록에 쓸 수 있습니다. 따라서 계산 전에 읽어들일 공유 메모리의 쓰기가 완료되었는지만 보장하면 되므로 한 번의 동기화만으로 충분합니다.

> 이중 버퍼는 읽기와 쓰기를 동시에 진행하게 하여 데이터 프리패치를 구현하고 메모리 지연을 숨깁니다.

이중 버퍼 기술을 적용하여 데이터 프리패치를 구현함으로써 계산 효율이 한층 더 향상되었습니다.

기본적으로 CUBLAS 공식 행렬 곱셈의 계산 효율에 근접할 수 있습니다.

```

```
