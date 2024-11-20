#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <curand_kernel.h>
#include <cuda_bf16.h> // __nv_bfloat16


struct KernelParams {
    size_t N;
    int threads;
    int blocks;
};

inline KernelParams get_kernel_params(const at::Tensor& tensor) {
    KernelParams params;
    params.N = tensor.numel();
    params.threads = 256;
    params.blocks = (params.N + params.threads - 1) / params.threads;
    return params;
}

inline void check_tensors(const at::Tensor& target, const at::Tensor& source) {
    TORCH_CHECK(target.is_cuda(), "Target tensor must be on CUDA device");
    TORCH_CHECK(source.is_cuda(), "Source tensor must be on CUDA device");
    TORCH_CHECK(target.dtype() == at::kBFloat16, "Target tensor must be bfloat16");
    TORCH_CHECK(source.dtype() == at::kFloat, "Source tensor must be float32");
    TORCH_CHECK(target.numel() == source.numel(), "Target and source must have the same number of elements");
}

inline void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}


// device function for stochastic rounding
__device__ __forceinline__ __nv_bfloat16 stochastic_rounding(
    float value,
    uint64_t idx,
    uint64_t seed
) {
    // Initialize curand state
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    // Convert float32 to uint32 representation
    uint32_t value_uint32 = __float_as_uint(value);

    // Generate random 16-bit integer
    uint32_t r = curand(&state) & 0xFFFF;  // Use &state

    // Add the random integer to the lower 16 bits of the mantissa
    value_uint32 += r;

    // Mask off the lower 16 bits
    value_uint32 &= 0xFFFF0000u;

    // Convert back to float32
    float rounded_float = __uint_as_float(value_uint32);

    // Convert float32 to bfloat16
    return __float2bfloat16(rounded_float);
}


__global__ void copy_stochastic_kernel(
    __nv_bfloat16* target,
    const float* source,
    size_t N,
    uint64_t seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    target[idx] = stochastic_rounding(source[idx], idx, seed);
}


// interface
void copy_stochastic(
    at::Tensor& target,
    const at::Tensor& source,
    uint64_t seed
) {
    check_tensors(target, source);
    KernelParams params = get_kernel_params(source);

    copy_stochastic_kernel<<<params.blocks, params.threads>>>(
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()),
        source.data_ptr<float>(),
        params.N,
        seed
    );

    check_cuda_error();
}


__global__ void add_stochastic_kernel(
    __nv_bfloat16* target,
    const float* source,
    size_t N,
    uint64_t seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // get a fp32 copy of the target for the operation
    float target_float = __bfloat162float(target[idx]);

    // in place to save
    target_float = target_float + source[idx];

    // write the fp32 result to bf16 target with stochastic rounding
    target[idx] = stochastic_rounding(target_float, idx, seed);
}


void add_stochastic(
    at::Tensor& target,
    const at::Tensor& source,
    uint64_t seed
) {
 
    check_tensors(target, source);
    KernelParams params = get_kernel_params(source);

    add_stochastic_kernel<<<params.blocks, params.threads>>>(
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()),
        source.data_ptr<float>(),
        params.N,
        seed
    );

    check_cuda_error();
}


__global__ void sub_stochastic_kernel(
    __nv_bfloat16* target,
    const float* source,
    size_t N,
    uint64_t seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // get a fp32 copy of the target for the operation
    float target_float = __bfloat162float(target[idx]);

    // in place to save
    target_float = target_float - source[idx];

    // write the fp32 result to bf16 target with stochastic rounding
    target[idx] = stochastic_rounding(target_float, idx, seed);
}

void sub_stochastic(
    at::Tensor& target,
    const at::Tensor& source,
    uint64_t seed
) {
    check_tensors(target, source);
    KernelParams params = get_kernel_params(source);

    sub_stochastic_kernel<<<params.blocks, params.threads>>>(
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()),
        source.data_ptr<float>(),
        params.N,
        seed
    );

    check_cuda_error();
}


__global__ void mul_stochastic_kernel(
    __nv_bfloat16* target,
    const float* source,
    size_t N,
    uint64_t seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // get a fp32 copy of the target for the operation
    float target_float = __bfloat162float(target[idx]);

    // in place to save
    target_float = target_float * source[idx];

    // write the fp32 result to bf16 target with stochastic rounding
    target[idx] = stochastic_rounding(target_float, idx, seed);
}


void mul_stochastic(
    at::Tensor& target,
    const at::Tensor& source,
    uint64_t seed
) {
    check_tensors(target, source);
    KernelParams params = get_kernel_params(source);

    mul_stochastic_kernel<<<params.blocks, params.threads>>>(
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()),
        source.data_ptr<float>(),
        params.N,
        seed
    );

    check_cuda_error();
}


__global__ void div_stochastic_kernel(
    __nv_bfloat16* target,
    const float* source,
    size_t N,
    uint64_t seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // get a fp32 copy of the target for the operation
    float target_float = __bfloat162float(target[idx]);

    // in place to save
    target_float = target_float / source[idx];

    // write the fp32 result to bf16 target with stochastic rounding
    target[idx] = stochastic_rounding(target_float, idx, seed);
}


void div_stochastic(
    at::Tensor& target,
    const at::Tensor& source,
    uint64_t seed
) {
    check_tensors(target, source);
    KernelParams params = get_kernel_params(source);

    div_stochastic_kernel<<<params.blocks, params.threads>>>(
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()),
        source.data_ptr<float>(),
        params.N,
        seed
    );

    check_cuda_error();
}


__global__ void lerp_stochastic_kernel(
    __nv_bfloat16* target,
    const float* source,
    const float weight,
    size_t N,
    uint64_t seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // get a fp32 copy of the target for the operation
    float target_float = __bfloat162float(target[idx]);

    // in place to save
    target_float = target_float * (1 - weight) + source[idx] * weight;

    // write the fp32 result to bf16 target with stochastic rounding
    target[idx] = stochastic_rounding(target_float, idx, seed);
}


void lerp_stochastic(
    at::Tensor& target,
    const at::Tensor& source,
    const float weight,
    uint64_t seed
) {
    check_tensors(target, source);
    KernelParams params = get_kernel_params(source);

    lerp_stochastic_kernel<<<params.blocks, params.threads>>>(
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()),
        source.data_ptr<float>(),
        weight,
        params.N,
        seed
    );

    check_cuda_error();
}

__global__ void addcmul_stochastic_kernel(
    __nv_bfloat16* target,
    const float* tensor1,
    const float* tensor2,
    const float value,
    size_t N,
    uint64_t seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // get a fp32 copy of the target for the operation
    float target_float = __bfloat162float(target[idx]);

    // in place to save
    target_float = target_float + tensor1[idx] * tensor2[idx] * value;

    // write the fp32 result to bf16 target with stochastic rounding
    target[idx] = stochastic_rounding(target_float, idx, seed);
}

void addcmul_stochastic(
    at::Tensor& target,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const float value,
    uint64_t seed
) {
    check_tensors(target, tensor1);
    check_tensors(target, tensor2);
    KernelParams params = get_kernel_params(tensor1);

    addcmul_stochastic_kernel<<<params.blocks, params.threads>>>(
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()),
        tensor1.data_ptr<float>(),
        tensor2.data_ptr<float>(),
        value,
        params.N,
        seed
    );

    check_cuda_error();
}


__global__ void addcdiv_stochastic_kernel(
    __nv_bfloat16* target,
    const float* tensor1,
    const float* tensor2,
    const float value,
    size_t N,
    uint64_t seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // get a fp32 copy of the target for the operation
    float target_float = __bfloat162float(target[idx]);

    // in place to save
    target_float = target_float + tensor1[idx] / tensor2[idx] * value;

    // write the fp32 result to bf16 target with stochastic rounding
    target[idx] = stochastic_rounding(target_float, idx, seed);
}


void addcdiv_stochastic(
    at::Tensor& target,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const float value,
    uint64_t seed
) {
    check_tensors(target, tensor1);
    check_tensors(target, tensor2);
    KernelParams params = get_kernel_params(tensor1);

    addcdiv_stochastic_kernel<<<params.blocks, params.threads>>>(
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()),
        tensor1.data_ptr<float>(),
        tensor2.data_ptr<float>(),
        value,
        params.N,
        seed
    );

    check_cuda_error();
}


// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_stochastic", &copy_stochastic, "Copy stochastic (CUDA)");
    m.def("add_stochastic", &add_stochastic, "Add stochastic (CUDA)");
    m.def("sub_stochastic", &sub_stochastic, "Subtract stochastic (CUDA)");
    m.def("mul_stochastic", &mul_stochastic, "Multiply stochastic (CUDA)");
    m.def("div_stochastic", &div_stochastic, "Divide stochastic (CUDA)");
    m.def("lerp_stochastic", &lerp_stochastic, "Linear interpolation stochastic (CUDA)");
    m.def("addcmul_stochastic", &addcmul_stochastic, "Addcmul stochastic (CUDA)");
    m.def("addcdiv_stochastic", &addcdiv_stochastic, "Addcdiv stochastic (CUDA)");
}
