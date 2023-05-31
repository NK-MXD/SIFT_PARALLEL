cmake_minimum_required(VERSION 3.22)
project( sift_parallel )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native -mavx2 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native -mavx2")

find_package(OpenCV REQUIRED) 
find_package(OpenMP)
# if (OpenMP_FOUND)
# 	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# 	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# 	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(sift_parallel main.cpp sift.cpp sift.h match.cpp match.h display.cpp display.h)

target_link_libraries(sift_parallel ${OpenCV_LIBS})
