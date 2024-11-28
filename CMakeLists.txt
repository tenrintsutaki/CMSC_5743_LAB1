cmake_minimum_required(VERSION 3.29)
project(LAB_1)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

add_executable(LAB_1 test.cpp
        winograd.cpp
        strassen.cpp
        npy_read_test.cpp
        lab3.cpp)

add_library(cnpy STATIC cnpy/cnpy.cpp)
target_link_libraries(LAB_1 PRIVATE cnpy)

# 查找 zlib 库
find_package(ZLIB REQUIRED)

# 将 zlib 链接到 cnpy 库和可执行文件
target_link_libraries(cnpy PRIVATE ZLIB::ZLIB)
target_link_libraries(LAB_1 PRIVATE cnpy)