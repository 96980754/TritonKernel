#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>

// 原始的kernel函数（正确的版本）
__global__ void matrix_multiplication_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K) {

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int index = tid + bid * block_size;
    int x = index % K;      // C的列索引 (0 到 K-1)
    int y = index / K;      // C的行索引 (0 到 M-1)

    if (x < K && y < M) {   // 注意：这里应该是 x < K, y < M
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            float e_x = A[y * N + i];   // A是M×N矩阵
            float e_y = B[i * K + x];   // B是N×K矩阵
            sum += e_x * e_y;
        }
        C[index] = sum;     // C是M×K矩阵，使用一维索引
    }
}

// CPU版本的矩阵乘法用于验证 (M×N * N×K = M×K)
void cpu_matrix_multiplication(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int N, int K) {
    
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < K; x++) {
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += A[y * N + i] * B[i * K + x];
            }
            C[y * K + x] = sum;  // C的宽度是K
        }
    }
}

class MatrixMultiplicationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化CUDA设备
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        cudaSetDevice(0);
    }

    void TearDown() override {
        // 确保所有CUDA操作完成
        cudaDeviceSynchronize();
    }

    // 辅助函数：比较GPU和CPU的结果
    bool compareResults(const std::vector<float>& gpu_result, 
                        const std::vector<float>& cpu_result, 
                        float tolerance = 1e-4) {
        if (gpu_result.size() != cpu_result.size()) {
            std::cout << "Size mismatch: GPU=" << gpu_result.size() 
                     << ", CPU=" << cpu_result.size() << std::endl;
            return false;
        }
        
        for (size_t i = 0; i < gpu_result.size(); i++) {
            if (std::fabs(gpu_result[i] - cpu_result[i]) > tolerance) {
                std::cout << "Mismatch at index " << i 
                         << ": GPU=" << gpu_result[i] 
                         << ", CPU=" << cpu_result[i] 
                         << ", diff=" << std::fabs(gpu_result[i] - cpu_result[i]) 
                         << std::endl;
                return false;
            }
        }
        return true;
    }

    // 辅助函数：生成随机矩阵
    std::vector<float> generateRandomMatrix(int rows, int cols, float min_val = -1.0f, float max_val = 1.0f) {
        std::vector<float> matrix(rows * cols);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        
        for (auto& val : matrix) {
            val = dis(gen);
        }
        return matrix;
    }

    // 辅助函数：运行矩阵乘法测试
    void runMatrixMultiplicationTest(int M, int N, int K, 
                                     const std::vector<float>& h_A,
                                     const std::vector<float>& h_B) {
        std::vector<float> h_C_gpu(M * K, 0.0f);
        std::vector<float> h_C_cpu(M * K, 0.0f);
        
        // 分配设备内存
        float *d_A, *d_B, *d_C;
        ASSERT_EQ(cudaMalloc(&d_A, M * N * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_B, N * K * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_C, M * K * sizeof(float)), cudaSuccess);
        
        // 拷贝数据到设备
        ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        
        // 启动kernel
        int total_threads = M * K;  // 注意：输出矩阵的元素个数是M*K
        int block_size = 256;
        int grid_size = (total_threads + block_size - 1) / block_size;
        
        matrix_multiplication_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
        
        // 检查kernel执行是否有错误
        cudaError_t kernel_error = cudaGetLastError();
        ASSERT_EQ(kernel_error, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(kernel_error);
        
        // 拷贝结果回主机
        ASSERT_EQ(cudaMemcpy(h_C_gpu.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        
        // CPU计算结果
        cpu_matrix_multiplication(h_A, h_B, h_C_cpu, M, N, K);
        
        // 比较结果
        EXPECT_TRUE(compareResults(h_C_gpu, h_C_cpu)) << "Matrix multiplication failed for M=" << M 
                                                       << ", N=" << N << ", K=" << K;
        
        // 清理
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};

// 测试1：基本功能测试 (2×3 * 3×4 = 2×4)
TEST_F(MatrixMultiplicationTest, BasicMultiplication) {
    int M = 2, N = 3, K = 4;
    
    // 初始化测试数据
    std::vector<float> h_A = {
        1.0f, 2.0f, 3.0f,  // 第0行
        4.0f, 5.0f, 6.0f   // 第1行
    };  // 2×3矩阵
    
    std::vector<float> h_B = {
        1.0f, 2.0f, 3.0f, 4.0f,   // 第0行
        5.0f, 6.0f, 7.0f, 8.0f,   // 第1行
        9.0f, 10.0f, 11.0f, 12.0f  // 第2行
    };  // 3×4矩阵
    
    runMatrixMultiplicationTest(M, N, K, h_A, h_B);
    
    // 手动验证几个值
    std::vector<float> h_C_cpu(M * K, 0.0f);
    cpu_matrix_multiplication(h_A, h_B, h_C_cpu, M, N, K);
    
    // C[0][0] = 1*1 + 2*5 + 3*9 = 1+10+27 = 38
    EXPECT_NEAR(h_C_cpu[0 * K + 0], 38.0f, 1e-4);
    // C[0][3] = 1*4 + 2*8 + 3*12 = 4+16+36 = 56
    EXPECT_NEAR(h_C_cpu[0 * K + 3], 56.0f, 1e-4);
    // C[1][2] = 4*3 + 5*7 + 6*11 = 12+35+66 = 113
    EXPECT_NEAR(h_C_cpu[1 * K + 2], 113.0f, 1e-4);
}

// 测试2：正方形矩阵 (4×4 * 4×4 = 4×4)
TEST_F(MatrixMultiplicationTest, SquareMatrixMultiplication) {
    int M = 4, N = 4, K = 4;
    auto h_A = generateRandomMatrix(M, N);
    auto h_B = generateRandomMatrix(N, K);
    runMatrixMultiplicationTest(M, N, K, h_A, h_B);
}

// 测试3：向量-矩阵乘法 (1×4 * 4×3 = 1×3)
TEST_F(MatrixMultiplicationTest, VectorMatrixMultiplication) {
    int M = 1, N = 4, K = 3;
    auto h_A = generateRandomMatrix(M, N);
    auto h_B = generateRandomMatrix(N, K);
    runMatrixMultiplicationTest(M, N, K, h_A, h_B);
}

// 测试4：矩阵-向量乘法 (3×4 * 4×1 = 3×1)
TEST_F(MatrixMultiplicationTest, MatrixVectorMultiplication) {
    int M = 3, N = 4, K = 1;
    auto h_A = generateRandomMatrix(M, N);
    auto h_B = generateRandomMatrix(N, K);
    runMatrixMultiplicationTest(M, N, K, h_A, h_B);
}

// 测试5：零矩阵乘法
TEST_F(MatrixMultiplicationTest, ZeroMatrixMultiplication) {
    int M = 3, N = 4, K = 5;
    std::vector<float> h_A(M * N, 0.0f);
    std::vector<float> h_B(N * K, 0.0f);
    runMatrixMultiplicationTest(M, N, K, h_A, h_B);
}

// 测试6：单位矩阵乘法 (3×3 * 3×3 = 3×3)
TEST_F(MatrixMultiplicationTest, IdentityMatrixMultiplication) {
    int M = 3, N = 3, K = 3;
    
    // 创建随机矩阵A
    auto h_A = generateRandomMatrix(M, N);
    
    // 创建单位矩阵B
    std::vector<float> h_B(N * K, 0.0f);
    for (int i = 0; i < N; i++) {
        h_B[i * K + i] = 1.0f;
    }
    
    // 结果应该等于A
    std::vector<float> h_C_gpu(M * K, 0.0f);
    std::vector<float> h_C_cpu(M * K, 0.0f);
    
    // ... (执行GPU和CPU计算)
    // 这里可以简化，直接比较结果与A
    runMatrixMultiplicationTest(M, N, K, h_A, h_B);
}

// 测试7：大矩阵乘法
TEST_F(MatrixMultiplicationTest, LargeMatrixMultiplication) {
    int M = 128, N = 256, K = 128;
    auto h_A = generateRandomMatrix(M, N, -10.0f, 10.0f);
    auto h_B = generateRandomMatrix(N, K, -10.0f, 10.0f);
    runMatrixMultiplicationTest(M, N, K, h_A, h_B);
}

// 测试8：单元素矩阵
TEST_F(MatrixMultiplicationTest, SingleElementMatrix) {
    int M = 1, N = 1, K = 1;
    std::vector<float> h_A = {3.0f};
    std::vector<float> h_B = {4.0f};
    runMatrixMultiplicationTest(M, N, K, h_A, h_B);
}

// 测试9：不同线程块大小
TEST_F(MatrixMultiplicationTest, DifferentBlockSizes) {
    int M = 16, N = 32, K = 16;
    auto h_A = generateRandomMatrix(M, N);
    auto h_B = generateRandomMatrix(N, K);
    
    std::vector<float> h_C_gpu(M * K, 0.0f);
    std::vector<float> h_C_cpu(M * K, 0.0f);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));
    
    cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
    
    // 测试不同块大小
    std::vector<int> block_sizes = {32, 64, 128, 256, 512};
    
    for (int block_size : block_sizes) {
        int grid_size = (M * K + block_size - 1) / block_size;
        
        // 清空输出
        cudaMemset(d_C, 0, M * K * sizeof(float));
        
        matrix_multiplication_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
        
        cudaMemcpy(h_C_gpu.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
        cpu_matrix_multiplication(h_A, h_B, h_C_cpu, M, N, K);
        
        EXPECT_TRUE(compareResults(h_C_gpu, h_C_cpu)) 
            << "Failed for block_size=" << block_size;
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 测试10：压力测试 - 非对称大矩阵
TEST_F(MatrixMultiplicationTest, StressTest) {
    int M = 512, N = 256, K = 1024;
    auto h_A = generateRandomMatrix(M, N, -100.0f, 100.0f);
    auto h_B = generateRandomMatrix(N, K, -100.0f, 100.0f);
    runMatrixMultiplicationTest(M, N, K, h_A, h_B);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // 检查CUDA设备
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable device found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // 显示设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;
    
    return RUN_ALL_TESTS();
}