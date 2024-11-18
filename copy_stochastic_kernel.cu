#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// hash function for per-thread random number generation
__device__ unsigned int hash(unsigned int x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

__global__ void copy_stochastic_kernel(float* target, const float* source, int N, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Reinterpret the float as int32
    int32_t src_int = __float_as_int(source[idx]);

    // Generate a random 16-bit integer in [0, 65535]
    uint32_t rand16 = hash(idx + seed) & 0xFFFF;

    // Add the random integer to the source
    uint32_t result_int = src_int + rand16;

    // Mask off the lower 16 bits
    result_int &= 0xFFFF0000;

    // Reinterpret as float and store in target
    target[idx] = __int_as_float(result_int);
}

void copy_stochastic_kernel_launcher(torch::Tensor target, torch::Tensor source, unsigned int seed) {
    const int threads = 1024;
    const int blocks = (source.numel() + threads - 1) / threads;
    int N = source.numel();

    copy_stochastic_kernel<<<blocks, threads>>>(
        target.data_ptr<float>(),
        source.data_ptr<float>(),
        N,
        seed);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in copy_stochastic_kernel: %s\n", cudaGetErrorString(err));
    }
}
