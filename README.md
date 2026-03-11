![](images/head.png)

![](https://img.shields.io/badge/build-passing-brightgreen) ![](https://img.shields.io/badge/ubuntu-18.04-blue) ![](https://img.shields.io/badge/cuda-10.2-blue) ![](https://img.shields.io/badge/nvidia-RTX3090-blue) ![](https://img.shields.io/badge/cmake-3.21-blue)

# 개요

NVIDIA GPU를 대상으로 CUDA SGEMM(단정밀도 행렬 곱) 성능을 단계적으로 최적화한 프로젝트입니다.

| 커널     | 설명                          | GFLOPS   | 사용자 커널/CUBLAS(%) |
| -------- | ----------------------------- | -------- | --------------------- |
| CUBLAS   | 공식 라이브러리               | 14448.69 | 기준                  |
| kernel_1 | 기본(naive) 구현              | 2262.168 | 15.65657              |
| kernel_2 | 공유 메모리 캐싱              | 4216.536 | 29.18283              |
| kernel_3 | 1차원 Thread Tile 최적화      | 7809.629 | 54.05078              |
| kernel_4 | 2차원 Thread Tile 최적화      | 12251.3  | 84.79179              |
| kernel_5 | 레지스터 캐싱                 | 12177.95 | 84.28412              |
| kernel_6 | FLOAT4 벡터화 메모리 접근     | 13161.49 | 91.09125              |
| kernel_7 | 더블 버퍼 기반 프리패치       | 13634.98 | 94.36832              |

> NVIDIA GeForce RTX 3090, 행렬 크기 5120

# 환경

- Ubuntu 18.04.5 LTS / gcc 7.5.0
- CUDA 10.2

# 디렉터리

```text
NVIDIA_SGEMM_PRACTICE                                   # 루트 디렉터리
    ├── images                                          # 결과 이미지
    ├── test                                            # 측정 결과 텍스트
    ├── src                                             # 소스 코드
    │    ├── kernel                                     # 커널 구현
    │    ├── kernel.cuh
    │    ├── utils.cuh                                  # 유틸 선언
    │    └── utils.cu                                   # 유틸 구현
    ├── plot.py                                         # test 결과 그래프 생성
    ├── run.sh                                          # 실행 스크립트
    ├── sgemm.cu                                        # 메인 프로그램
    └── CMakeLists.txt                                  # 빌드 설정
```

# 실행 방법

1. NVCC 아키텍처 설정
   - `CMakeLists.txt`에서 `CUDA_NVCC_FLAGS` 값을 GPU 환경에 맞게 수정
2. 최대 행렬 크기 설정
   - `sgemm.cu`의 `size_len` 값을 조정 (처음에는 작은 값 권장)
3. 빌드
   - `cd build && cmake .. && make`
4. 실행
   - `./run.sh` (결과는 `test/`에 저장)
5. 그래프 생성
   - 예: `python plot.py 0 1` (CUBLAS vs kernel_1)

# 단계별 최적화 요약

## kernel_1: 기본 구현

각 스레드가 C 행렬의 원소 하나를 계산합니다. 구현은 단순하지만 전역 메모리 접근이 많아 성능이 낮습니다.

![](./images/describe_kernel_1.png)

![](./images/kernel_culas_vs_1.png)

## kernel_2: 공유 메모리 캐싱

A/B 타일을 공유 메모리에 올려 전역 메모리 재접근을 줄입니다. 대역폭 병목 완화에 효과적입니다.

![](./images/describe_kernel_2.png)
![](./images/kernel_1_vs_2.png)

## kernel_3: 1D Thread Tile

블록 크기와 BK를 재조정하고, 하나의 스레드가 여러 출력 원소를 계산하도록 변경해 계산/메모리 비율을 높입니다.

## kernel_4: 2D Thread Tile

스레드 타일을 2차원으로 확장해 데이터 재사용을 더욱 늘리고 성능을 크게 향상합니다.

## kernel_5: 레지스터 캐싱

공유 메모리에서 반복 접근하는 값을 레지스터로 캐시해 공유 메모리 접근 지연을 줄입니다.

## kernel_6: FLOAT4 벡터화

벡터 로드/스토어를 활용해 메모리 명령 수와 주소 계산 오버헤드를 줄여 처리량을 개선합니다.

## kernel_7: 더블 버퍼 프리패치

읽기/쓰기 버퍼를 분리해 다음 타일을 미리 가져오며 연산과 메모리 접근을 겹쳐 지연을 숨깁니다.

최종적으로 CUBLAS에 근접한 성능을 달성합니다.
