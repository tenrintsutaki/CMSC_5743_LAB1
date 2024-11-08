cmake_minimum_required(VERSION 3.29)
project(LAB_1)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

add_executable(LAB_1 test.cpp
        img2col.cpp
        strassen.cpp)

