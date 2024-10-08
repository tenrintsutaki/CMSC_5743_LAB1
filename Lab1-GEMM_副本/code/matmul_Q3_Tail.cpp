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

constexpr int n = 1024;
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

void matmul_unrolling() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k+=2) {
                C[i][j] += A[i][k] * B[k][j];
                if (k + 1 < n){
                    C[i][j] += A[i][k+1] * B[k+1][j];
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

void matmul_ikj_unrolling() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k+=2) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][k] * B[k][j];
                if (k + 1 < n){
                    C[i][j] += A[i][k + 1] * B[k + 1][j];
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

void matmul_AT_unrolling() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AT[i][j] = A[j][i];
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j+=2) {
            for (int k = 0; k < n; k++) {
                C[i][j] += AT[k][i] * B[k][j];
                if (j + 1 < n){
                    C[i][j + 1] += AT[k][i] * B[k][j + 1];
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

void matmul_BT_unrolling() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      BT[i][j] = B[j][i];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j+=2) {
      for (int k = 0; k < n; k++) {
          C[i][j] += A[i][k] * BT[j][k];
          if (j + 1 < n){
              C[i][j+1] += A[i][k] * BT[j+1][k];
          }
      }   
    }
  }
}

void experiment_ijk(){
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
        matmul_ikj_unrolling();
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for matmul_IJK Unrolling Calculation: %f\n", avg_time / 32);
}

void experiment_matmul(){
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
        matmul_unrolling();
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for matmul Unrolling Calculation: %f\n", avg_time / 32);
}

void experiment_AT(){
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
        matmul_AT_unrolling();
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for AT Unrolling Calculation: %f\n", avg_time / 32);
}

void experiment_BT(){
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
        matmul_BT_unrolling();
        test();
        avg_time += get_time() - t;
    }
    printf("Avg Time for BT Unrolling Calculation: %f\n", avg_time / 32);
}

int main() {
    init();
    experiment_matmul();
    experiment_ijk();
    experiment_AT();
    experiment_BT();
  return 0;
}

