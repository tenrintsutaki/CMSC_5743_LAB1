//
// Created by Tsutaki Tenrin on 24-11-28.
//
#include <iostream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include "cnpy/cnpy.h" // 用于加载 .npy 文件

using namespace std;

// 定义稀疏点结构
struct SparsePoint {
    int batch; // 批次索引 (batch)
    int x;     // 高度位置 (row)
    int y;     // 宽度位置 (column)
    vector<float> features; // 输入通道的特征值
};

// 定义卷积核的结构
struct Kernel {
    int kernel_size; // 卷积核大小 (例如 3 表示 3x3)
    vector<pair<int, int>> offsets; // 卷积核的偏移
};

// 创建卷积核的偏移
Kernel createKernel(int kernel_size) {
    Kernel kernel;
    kernel.kernel_size = kernel_size;
    int radius = kernel_size / 2; // 半径 (3x3的半径是1)
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            kernel.offsets.emplace_back(dx, dy);
        }
    }
    return kernel;
}

// Rulebook 的创建
unordered_map<int, vector<pair<int, int>>> createRulebook(
    const vector<SparsePoint>& inputPoints,
    const Kernel& kernel,
    int height,
    int width
) {
    unordered_map<int, vector<pair<int, int>>> rulebook;

    for (size_t i = 0; i < inputPoints.size(); ++i) {
        const auto& point = inputPoints[i];
        for (const auto& offset : kernel.offsets) {
            int nx = point.x + offset.first;
            int ny = point.y + offset.second;

            // 确保邻域坐标在矩阵范围内
            if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
                // 计算输出点的哈希值作为键
                int hashKey = nx * width + ny;
                rulebook[hashKey].emplace_back(i, hashKey);
            }
        }
    }
    return rulebook;
}

// Submanifold Sparse Convolution 操作
vector<SparsePoint> submSparseConv(
    const vector<SparsePoint>& inputPoints,
    const Kernel& kernel,
    const unordered_map<int, vector<pair<int, int>>>& rulebook,
    const vector<vector<float>>& weights, // 卷积权重 (in_channels x out_channels)
    int out_channels
) {
    vector<SparsePoint> outputPoints;

    // 遍历 Rulebook，执行卷积操作
    for (const auto& rule : rulebook) {
        int outputHashKey = rule.first; // 输出点的哈希值
        const auto& mappings = rule.second;

        vector<float> newFeatures(out_channels, 0.0f); // 输出点的特征值 (初始化为0)

        for (const auto& mapping : mappings) {
            int inputIndex = mapping.first; // 输入点索引
            const auto& inputFeatures = inputPoints[inputIndex].features;

            // 计算卷积操作 (输入特征 x 权重)
            for (size_t in_ch = 0; in_ch < inputFeatures.size(); ++in_ch) {
                for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
                    newFeatures[out_ch] += inputFeatures[in_ch] * weights[in_ch][out_ch];
                }
            }
        }

        // 还原输出点的坐标
        int x = outputHashKey / kernel.offsets.size();
        int y = outputHashKey % kernel.offsets.size();

        // 添加到输出点
        outputPoints.push_back({0, x, y, newFeatures});
    }

    return outputPoints;
}

// 从 .npy 文件中加载稀疏矩阵
vector<SparsePoint> loadSparseMatrix(const string& filePath, int height, int width, int in_channels) {
    // 加载 .npy 文件
    cnpy::NpyArray arr = cnpy::npy_load(filePath);
    double* data = arr.data<double>();

    vector<SparsePoint> sparsePoints;

    // 遍历矩阵，提取非零值
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            double value = data[i * width + j];
            if (value != 0.0) { // 如果矩阵的值非零，则保存为稀疏点
                SparsePoint point;
                point.batch = 0; // 假设 batch = 1
                point.x = i;
                point.y = j;
                point.features.push_back(static_cast<float>(value)); // 单通道输入
                sparsePoints.push_back(point);
            }
        }
    }

    return sparsePoints;
}

// 主程序
int main() {
    // 输入矩阵参数
    int height = 64, width = 4096, in_channels = 1;
    int out_channels = 256; // 输出通道数

    // 从 .npy 文件加载稀疏矩阵
    string filePath = "../pointcloud.npy";
    vector<SparsePoint> inputPoints = loadSparseMatrix(filePath, height, width, in_channels);
    Kernel kernel = createKernel(3); // 卷积核大小为 3x3
    vector<vector<float>> weights(in_channels, vector<float>(out_channels, 1.0f)); // 简单初始化为 1.0
    unordered_map<int, vector<pair<int, int>>> rulebook = createRulebook(inputPoints, kernel, height, width);
    vector<SparsePoint> outputPoints = submSparseConv(inputPoints, kernel, rulebook, weights, out_channels);

    // 输出结果
    cout << "Output Sparse Points:" << endl;
    for (const auto& point : outputPoints) {
        cout << "Batch: " << point.batch << ", Position (" << point.x << ", " << point.y << "), Features: ";
        for (float feature : point.features) {
            cout << feature << " ";
        }
        cout << endl;
    }

    return 0;
}