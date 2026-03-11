cmake_minimum_required(VERSION 3.0)
project(NVIDIA_SGEMM_PRACTICE)

# gcc/g++ 컴파일 옵션 설명:
# -O1~O3: 컴파일러 최적화 단계. 단계가 높을수록 최적화 효과가 커지지만 컴파일 시간은 늘어납니다.
# -std=c++11: C++11 표준으로 컴파일
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

# nvcc 컴파일 옵션 설명:
# -g: 호스트 코드에 디버그 정보를 추가
# -G: 디바이스 코드 디버그 정보 생성(대부분 최적화 비활성화로 실행 속도 저하)
# -Xptxas -dlcm=ca 는 L1 캐시 사용, -Xptxas -dlcm=cg 는 L1 캐시 비사용

# set(CUDA_NVCC_FLAGS -g;-G;-Xptxas;-dlcm=ca)
# set(CUDA_NVCC_FLAGS -Xptxas;-dlcm=cg)
set(CUDA_NVCC_FLAGS -arch=compute_70;-code=compute_70)

# FIND CUDA 오류가 나면 ~/.bashrc에 환경 변수와 동적 라이브러리 경로를 추가하세요
# CUDA_HOME=/usr/local/cuda
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
find_package(CUDA REQUIRED)

# 헤더 파일 검색 경로 설정
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# 컴파일할 소스 파일 경로 설정
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)
# 실행 파일 출력 경로
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# 실행 파일 생성
CUDA_ADD_EXECUTABLE(sgemm sgemm.cu ${SRC})

# link cudart cublas
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

