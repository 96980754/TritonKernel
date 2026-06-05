#include <cuda_runtime.h>

// input, output are device pointers
__device__ float warpReduceSum(float val){
    for(int i = warpSize / 2;i>=1;i>>=1){
        val += __shfl_down_sync(0xffffffff,val,i);
    }
    return val;
}
__global__ void reduction(const float* input, float* output, int N){
    extern __shared__ float shared[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int wid = tid / 32;
    int lane = tid % 32;
    int num_warps = block_size / 32;
    int index = tid + block_size * bid;

    // if (index < N) {
    //     shared[tid] = input[index];
    // } else {
    //     shared[tid] = 0.0f;
    // }
    // __syncthreads();

    float val;
    if (index < N) {
        val = input[index];
    } else {
        val = 0.0f;
    }
    __syncthreads();

    val = warpReduceSum(val);
    if (lane == 0){shared[wid] = val;}

    __syncthreads();
    for(int i = num_warps/2 ;i>=1 ;i>>=1){
        if(tid < i){
            shared[tid] += tid+i<num_warps ? shared[tid+i] : 0.0f;
        }
    }
    __syncthreads();
    if(tid == 0){
        atomicAdd(output,shared[0]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // size_t sharedMemSize = threadsPerBlock * sizeof(float);
    size_t sharedMemSize = (threadsPerBlock/32) * sizeof(float);

    reduction<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(input, output, N);
    cudaDeviceSynchronize();
}

