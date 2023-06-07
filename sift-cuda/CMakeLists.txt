cmake_minimum_required(VERSION 3.25)
project(sift_cuda CUDA CXX)

# -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda-11.8/targets/x86_64-linux/include)  # texture_types.h

add_executable(sift_cuda main.cu sift_cuda.cuh sift_cuda.cu img_cuda.cuh util_cuda.cuh)

target_link_libraries(sift_cuda ${OpenCV_LIBS})
set_target_properties(sift_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
