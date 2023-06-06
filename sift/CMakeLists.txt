# set( /work/home/acmhsiv3ds/usr)
cmake_minimum_required(VERSION 3.23)
project(sift)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx -mavx2 -fopenmp -msse -msse2")

# Note: manually set `OpenCV_DIR` in Windows environment
# At directory `build`, run `cmake -DCMAKE_MAKE_PROGRAM="C:/Program Files/JetBrains/CLion 2023.1.3/bin/ninja/win/x64/ninja.exe" -G Ninja -S ..`,
# then run `cmake build .` to build the executable file
set(OpenCV_DIR /work/home/acmhsiv3ds/opencv-3.4.16/build)
find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(sift main.cpp sift.h sift.cpp demo/test_match.cpp demo/test_omp.cpp demo/test_multithreading.cpp demo/mtest_multithreadandSIMD.cpp demo/mtest_simd.cpp)

target_link_libraries(sift ${OpenCV_LIBS})