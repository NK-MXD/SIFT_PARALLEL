cmake_minimum_required(VERSION 3.25)
project(test_image)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")

set(OpenCV_DIR C:/opencv/build/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test_image main.cpp ../../util.h ../../image.h)

target_link_libraries(test_image ${OpenCV_LIBS})
