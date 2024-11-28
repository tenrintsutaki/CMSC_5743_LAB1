//
// Created by Tsutaki Tenrin on 24-11-28.
//

#include "cnpy/cnpy.h"
#include <iostream>
#include <vector>

// std::vector <std::string> read_file(std::string file_name) {}
//
// int main() {
//     cnpy::NpyArray arr = cnpy::npy_load("../pointcloud.npy");
//     double* data = arr.data<double>();
//
//     // 获取数组形状
//     std::vector<size_t> shape = arr.shape;
//
//     // 打印数组形状
//     std::cout << "Shape: ";
//     for (size_t dim : shape) {
//         std::cout << dim << " ";
//     }
//     std::cout << std::endl;
//
//     // 打印数组内容
//     size_t total_elements = 1;
//     for (size_t dim : shape) total_elements *= dim;
//
//     std::cout << "Data: ";
//     for (size_t i = 0; i < total_elements; ++i) {
//         std::cout << data[i] << " ";
//     }
//     std::cout << std::endl;
//
//     return 0;
// }
