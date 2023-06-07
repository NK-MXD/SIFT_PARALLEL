cmake_minimum_required(VERSION 3.22)
project( sift_parallel )

set(OpenCV_DIR "D:/softwareSpace/openCV/opencv/build/x64/vc16/lib")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -mavx -fopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -march=native -mavx -mavx2")

find_package(OpenCV REQUIRED) 
find_package(OpenMP)
if (OpenMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(OpenCV_INCLUDE_DIRS D:/softwareSpace/openCV/opencv/source/include)
include_directories(${OpenCV_INCLUDE_DIRS})

#add_executable(sift_parallel main.cpp sift.cpp sift.h match.cpp match.h display.cpp display.h)
add_executable(sift_parallel main.cpp sift.cpp sift.h)

target_link_libraries(sift_parallel ${OpenCV_LIBS})
