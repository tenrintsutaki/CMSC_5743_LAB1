// g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall && ./matmul

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int n = 256;
int A[n][n];
int B[n][n];
int BT[n][n];
int AT[n][n];
int C[n][n];
int C_groundtruth[n][n];

void init() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = rand(); 
      B[i][j] = rand(); 
    } 
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C_groundtruth[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void test() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      assert(C[i][j] == C_groundtruth[i][j]);
    }
  }
}

void matmul() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }   
    }
  }
}

void matmul_tiling(int size) {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i += size) {
        for (int j = 0; j < n; j += size) {
            for (int k = 0; k < n; k += size) {
                int maxM = std::min(i + size, n);
                int maxN = std::min(j + size, n);
                int maxP = std::min(k + size, n);
                for (int m = i; m < maxM; ++m){
                    for (int n = j; n < maxN; ++n){
                        for (int p = k; p < maxP; ++p){
                            C[m][p] += A[m][n] * B[n][p];
                        }
                    }
                }
            }
        }
    }
}

void matmul_ikj() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < n; j++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmul_ikj_tiling(int size) {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i += size) {
        for (int k = 0; k < n; k += size) {
            for (int j = 0; j < n; j += size) {
                int maxM = std::min(i + size, n);
                int maxN = std::min(j + size, n);
                int maxP = std::min(k + size, n);
                for (int m = i; m < maxM; ++m){
                    for (int p = k; p < maxP; ++p){
                        for (int n = j; n < maxN; ++n){
                            C[m][n] += A[m][p] * B[p][n];
                        }
                    }
                }
            }
        }
    }
}

void matmul_AT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      AT[i][j] = A[j][i];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += AT[k][i] * B[k][j];    
      }   
    }
  }
}

void matmul_AT_tiling(int size) {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AT[i][j] = A[j][i];
        }
    }
    for (int i = 0; i < n; i += size) {
        for (int j = 0; j < n; j += size) {
            for (int k = 0; k < n; k += size) {
                int maxM = std::min(i + size, n);
                int maxN = std::min(j + size, n);
                int maxP = std::min(k + size, n);
                for (int m = i; m < maxM; ++m){
                    for (int n = j; n < maxN; ++n){
                        for (int p = k; p < maxP; ++p){
                            C[m][n] += AT[p][m] * B[p][n];
                        }
                    }
                }
            }
        }
    }
}

void matmul_BT() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            BT[i][j] = B[j][i];
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * BT[j][k];
            }
        }
    }
}

void matmul_BT_tiling(int size) {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      BT[i][j] = B[j][i];
    }
  }
    for (int i = 0; i < n; i += size) {
        for (int j = 0; j < n; j += size) {
            for (int k = 0; k < n; k += size) {
                int maxM = std::min(i + size, n);
                int maxN = std::min(j + size, n);
                int maxP = std::min(k + size, n);
                for (int m = i; m < maxM; ++m){
                    for (int n = j; n < maxN; ++n){
                        for (int p = k; p < maxP; ++p){
                            C[m][n] += A[m][p] * BT[n][p];
                        }
                    }
                }
            }
        }
    }
}

void experiment_ijk(int size){
    float avg_time = 0.0f;
    for (int K = 0; K < 32; K++) {
        auto t = get_time();
        matmul_ikj();
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for matmul_IJK Calculation: %f\n", avg_time / 32);

    avg_time = 0.0f;
    for (int K = 0; K < 32; K++) {
        auto t = get_time();
        matmul_ikj_tiling(size);
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for matmul_IJK tiling Calculation: %f\n", avg_time / 32);
}

void experiment_matmul(int size){
    float avg_time = 0.0f;
    for (int K = 0; K < 32; K++) {
        auto t = get_time();
        matmul();
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for matmul Calculation: %f\n", avg_time / 32);

    avg_time = 0.0f;
    for (int K = 0; K < 32; K++) {
        auto t = get_time();
        matmul_tiling(size);
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for matmul tiling Calculation: %f\n", avg_time / 32);
}

void experiment_AT(int size){
    float avg_time = 0.0f;
    for (int K = 0; K < 32; K++) {
        auto t = get_time();
        matmul_AT();
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for AT Calculation: %f\n", avg_time / 32);

    avg_time = 0.0f;
    for (int K = 0; K < 32; K++) {
        auto t = get_time();
        matmul_AT_tiling(size);
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for AT tiling Calculation: %f\n", avg_time / 32);
}

void experiment_BT(int size){
    float avg_time = 0.0f;
    for (int K = 0; K < 32; K++) {
        auto t = get_time();
        matmul_BT();
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for BT Calculation: %f\n", avg_time / 32);

    avg_time = 0.0f;
    for (int K = 0; K < 32; K++) {
        auto t = get_time();
        matmul_BT_tiling(size);
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for BT tiling Calculation: %f\n", avg_time / 32);
}

int main() {
  init();
    experiment_matmul(32);
    experiment_ijk(32);
    experiment_AT(32);
    experiment_BT(32);
  return 0;
}

