//
// Created by 96980 on 2026/7/29.
//
#include<cuda_runtime.h>
#include<cmath>
__global__ void SoftmaxKernelV1(int n,int c,const float* input,float* output) {
    int bid = blockIdx.x;
    if (bid >= n) {
        return;
    }
    int tid = threadIdx.x;
    int bsize = blockDim.x;
    extern __shared__ float shared[];
    const float* x = input + bid*c;
    float* y = output + bid*c;
    //reduce 求行最大值
    float localmax = -INFINITY;
    for (int i=tid;i<c;i+=bsize) {
        localmax = fmaxf(x[i],localmax);
    }
    __syncthreads();
    shared[tid] = localmax;
    // block内部reduce
    for(int i = bsize/2;i>0;i>>=1) {
        __syncthreads();
        if (tid < i){
            shared[tid] = fmaxf(shared[tid],shared[i+tid]);
        }
    }
    __syncthreads();
    float row_max = shared[0];
    //求分子 //求和
    for (int i=tid;i<c;i+=bsize) {
        y[i] = expf(x[i]-row_max);
    }
    __syncthreads();
    float row_sum = 0.0f;
    for (int i=tid;i<c;i+=bsize) {
        row_sum += x[i];
    }
    __syncthreads();
    for (int i=tid;i<c;i+=bsize) {
        y[i] /= row_sum;
    }

}