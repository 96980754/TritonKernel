#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
// v1版本，采用共享内存 
template<const int BLOCK_SIZE>
__global__ void matrix_multiplication_kernel(
    const float* A, const float* B, float* C, 
    int M, int N,int K) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        const int BM = BLOCK_SIZE;
        const int BN = BLOCK_SIZE;
        const int BK = BLOCK_SIZE;

        int x = bx * BK + tx;  
        int y = by * BM + ty;

        __shared__ float sharedA[BM*BN];
        __shared__ float sharedB[BN*BK];
        float tmp = 0.0f;
        for(int i=0;i<N;i+=BN){
            if (y < M  && (i+tx) < N){
                sharedA[ty*BN + tx] = A[y*N + i+ tx];
            }else{
                sharedA[ty*BN + tx] = 0.0f;
            }

            if ((i+ty) < N && x < K){
                sharedB[ty*BK + tx] = B[(i+ty)*K + x];
            }else{
                sharedB[ty*BK + tx] = 0.0f;
            }
            __syncthreads();

            for(int j = 0;j<BN;j++){
                tmp += sharedA[ty*BN + j] * sharedB[j*BK + tx];
            }
            __syncthreads();
        }
        if(x < K && y < M){
            C[y*K + x] = tmp;
            // alpha * tmp + C[y*K + x]*beta
        }
}
int main(){
    float A[] = {1.0,2.0, 3.0, 4.0};
    float B[] = {5.0, 6.0, 7.0, 8.0};
    float C[] = {0.0, 0.0, 0.0, 0.0};

    float *d_A, *d_B, *d_C;
    int M = 2, N = 2, K = 2;
    cudaMalloc(&d_A, M*N*sizeof(float));
    cudaMalloc(&d_B, N*K*sizeof(float));
    cudaMalloc(&d_C, M*K*sizeof(float));
    cudaMemcpy(d_A, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M*K*sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x
                    , (M + blockSize.y - 1) / blockSize.y);
    matrix_multiplication_kernel<16><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
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
