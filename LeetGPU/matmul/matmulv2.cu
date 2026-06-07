
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
// v1版本，采用共享内存 


template<const int BLOCK_SIZE_M = 64, const int BLOCK_SIZE_N = 64, const int BLOCK_SIZE_K = 32,
         const int TM = 8, const int TN = 8>
__global__ void matmul_kernel(
    float *A, float *B, float *C,
    int M, int K, int N)
{
    const int BM = BLOCK_SIZE_M;
    const int BN = BLOCK_SIZE_N;
    const int BK = BLOCK_SIZE_K;
    // Block 索引
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // 线程在 block 内的索引
    int tid = threadIdx.x;
    int tx = (tid % (BN / TN)) * TN;
    int ty = (tid / (BN / TN)) * TM;
    int num_float4_A = BK / 4;  // 每行 BK/4 个 float4
    int num_float4_B = BN / 4;  // 每行 BN/4 个 float4

    int ay_float4 = tid / num_float4_A;
    int ax_float4 = tid % num_float4_A;

    int by_float4 = tid / num_float4_B;
    int bx_float4 = tid % num_float4_B;

    // 共享内存
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    // 全局矩阵移动到当前 Block 的起始位置
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // 累加器
    float accum[TM][TN] = {0.0f};

    // 主循环
    for (int k_block = 0; k_block < K; k_block += BK)
    {
        // 加载 A 到 sA (float4 向量化)
        int ay = ay_float4;
        int ax = ax_float4;
        if ( ay+bx*BM  < M && (k_block + ax * 4 + 3) < K)
        {
            // 边界安全，直接 float4 加载
            float4 tmp = *reinterpret_cast<const float4*>(&A[ay * K + k_block + ax * 4]);
            *reinterpret_cast<float4*>(&sA[ay * BK + ax * 4]) = tmp;
        }
        else if (ay+bx*BM < M)
        {
            // 逐元素加载并 padding 0
            for (int i = 0; i < 4; i++)
            {
                int a_col = ax * 4 + i;
                if (a_col < BK && (k_block + a_col) < K)
                    sA[ay * BK + a_col] = A[ay * K + k_block + a_col];
                else
                    sA[ay * BK + a_col] = 0.0f;
            }
        }

        // 加载 B 到 sB (float4 向量化)
        int _by = by_float4;
        int _bx = bx_float4;
        if (_by+BK*by < K && (_bx * 4 + 3) < N)
        {
            float4 tmp = *reinterpret_cast<const float4*>(&B[(k_block + _by) * N + _bx * 4]);
            *reinterpret_cast<float4*>(&sB[_by * BN + _bx * 4]) = tmp;
        }
        else if (_by+BK*by < K)
        {
            for (int i = 0; i < 4; i++)
            {
                int b_col = _bx * 4 + i;
                if (b_col < BN)
                    sB[_by * BN + b_col] = B[(k_block + _by) * N + b_col];
                else
                    sB[_by * BN + b_col] = 0.0f;
            }
        }

        __syncthreads();

        // 计算阶段
        float frag_a[TM];
        float frag_b[TN];

        for (int j = 0; j < BK; j++)
        {
            // 从 sA 取一列数据 (复用 TM 次)
            #pragma unroll
            for (int m = 0; m < TM; m++)
            {
                frag_a[m] = sA[(ty + m) * BK + j];
            }
            // 从 sB 取一行数据 (复用 TN 次)
            #pragma unroll
            for (int n = 0; n < TN; n++)
            {
                frag_b[n] = sB[j * BN + (tx + n)];
            }

            #pragma unroll
            for (int m = 0; m < TM; m++)
            {
                #pragma unroll
                for (int n = 0; n < TN; n++)
                {
                    accum[m][n] += frag_a[m] * frag_b[n];
                }
            }
        }

        __syncthreads();
    }

    // 写回 C
    #pragma unroll
    for (int m = 0; m < TM; m++)
    {
        #pragma unroll
        for (int n = 0; n < TN; n++)
        {
            int global_row = by * BM + ty + m;
            int global_col = bx * BN + tx + n;
            if (global_row < M && global_col < N)
            {
                C[(ty + m) * N + (tx + n)] = accum[m][n];
            }
        }
    }
}

int main(){
    float A[] = {1.0,2.0, 3.0, 4.0, 6.0, 7.0, 6.0, 7.0,2.0, 3.0, 4.0, 6.0};
    float B[] = {5.0, 6.0, 7.0, 8.0,2.0, 3.0, 4.0, 6.0, 4.0, 6.0, 7.0, 6.0};
    float C[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,};
    // 57 56 49 
    // 152 135 129 
    // 86 84 75 
    float *d_A, *d_B, *d_C;
    int M = 3, N = 4, K = 3;
    cudaMalloc(&d_A, M*N*sizeof(float));
    cudaMalloc(&d_B, N*K*sizeof(float));
    cudaMalloc(&d_C, M*K*sizeof(float));
    cudaMemcpy(d_A, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M*K*sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockSize(128);
    dim3 gridSize((K + 128 - 1) / 128
                    , (M + 128 - 1) / 128);
    matmul_kernel<128, 128, 8, 8, 8><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(C, d_C, M*K*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            std::cout<<C[i*K + j]<<" "; 
        }
        std::cout<<std::endl;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
