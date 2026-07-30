//
// Created by 96980 on 2026/7/29.
//

#include<cuda_runtime.h>
#include<cmath>
__device__ warpReduceSum(float val) {
    int lane = threadIdx.x % 32;
    //32是warpSize
    for (int i = 32/2; i > 0;i >>=1) {
        if (lane < i) {
            val += __shfl_down_sync(0xffffffff,val,i)
        }
    }
}
__device__ warpReduceMax(float val) {
    int lane = threadIdx.x % 32;
    for (int i = 32/2; i > 0;i >>=1) {
        if (lane < i) {
            val = fmaxf(val,__shfl_down_sync(0xffffffff,val,i))
        }
    }
}

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
    float val = x[tid];
    float localmax = warpReduceSum(val);
    if (tid % 32 ==0) {
        shared[bid]
    }
}