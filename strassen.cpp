#include <iostream>
#include <vector>
#include <sys/time.h>

using namespace std;

typedef vector<vector<int>> Matrix;

constexpr int matrix_size = 1024;
Matrix matrix_A(matrix_size, vector<int>(matrix_size));
Matrix matrix_B(matrix_size, vector<int>(matrix_size));

// double get_time() {
//     struct timeval tv;
//     gettimeofday(&tv, nullptr);
//     return tv.tv_sec + 1e-6 * tv.tv_usec;
// }

Matrix generateRandomMatrix(int size) {
    srand(static_cast<unsigned int>(time(0)));
    Matrix matrix(size, vector<int>(size)); // 2D -  vector
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = rand(); // Randomized Matrix
        }
    }

    return matrix;
}

void init(int size)
{
    matrix_A = generateRandomMatrix(size);
    matrix_B = generateRandomMatrix(size);

}

Matrix MatrixAdd(Matrix& A, Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] + B[i][j]; // Matrix A + Matrix B
    return C;
}

Matrix MatrixSubtract(Matrix& A, Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] - B[i][j]; // Matrix A - Matrix B
    return C;
}

Matrix StrassenAlgorithm(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n == 1) {
        return Matrix{{A[0][0] * B[0][0]}};
    }
    int divide = n / 2;
    Matrix A11(divide, vector<int>(divide)), A12(divide, vector<int>(divide));
    Matrix A21(divide, vector<int>(divide)), A22(divide, vector<int>(divide));
    Matrix B11(divide, vector<int>(divide)), B12(divide, vector<int>(divide));
    Matrix B21(divide, vector<int>(divide)), B22(divide, vector<int>(divide));

    for (int i = 0; i < divide; ++i) { // Calculate A11 to B22
        for (int j = 0; j < divide; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + divide];
            A21[i][j] = A[i + divide][j];
            A22[i][j] = A[i + divide][j + divide];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + divide];
            B21[i][j] = B[i + divide][j];
            B22[i][j] = B[i + divide][j + divide];
        }
    }
    // Calculate S1 to S7
    Matrix S1 = StrassenAlgorithm(MatrixSubtract(A12, A22), MatrixAdd(B21, B22));
    Matrix S2 = StrassenAlgorithm(MatrixAdd(A11, A22), MatrixAdd(B11, B22));
    Matrix S3 = StrassenAlgorithm(MatrixSubtract(A11, A21), MatrixAdd(B11, B12));
    Matrix S4 = StrassenAlgorithm(MatrixAdd(A11, A12), B22);
    Matrix S5 = StrassenAlgorithm(A11, MatrixSubtract(B12, B22));
    Matrix S6 = StrassenAlgorithm(A22, MatrixSubtract(B21, B11));
    Matrix S7 = StrassenAlgorithm(MatrixAdd(A21, A22), B11);

    Matrix Result(n, vector<int>(n));
    for (int i = 0; i < divide; ++i) { // Update the result by formula
        for (int j = 0; j < divide; ++j) {
            Result[i][j] = S1[i][j] + S2[i][j] - S4[i][j] + S6[i][j];
            Result[i][j + divide] = S5[i][j] + S4[i][j];
            Result[i + divide][j] = S7[i][j] + S6[i][j];
            Result[i + divide][j + divide] = S2[i][j] - S3[i][j] + S5[i][j] - S7[i][j];
        }
    }

    return Result;
}

void printMatrix(const Matrix& mat) {
    for (const auto& row : mat) {
        for (const auto& val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

// int main()
// {
//     init(matrix_size);
//     float avg_time = 0.0f;
//     for (int K = 0; K < 5; K++) {
//         auto t = get_time();
//         StrassenAlgorithm(matrix_A, matrix_B);
//         avg_time += get_time() - t;
//     }
//     printf("Avg Time for Calculation Strassen: %f for n size %d \n", avg_time / 5, matrix_size);
//     return 0;
// }