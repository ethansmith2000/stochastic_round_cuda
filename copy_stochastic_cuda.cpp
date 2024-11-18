#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

void copy_stochastic_kernel_launcher(torch::Tensor target, torch::Tensor source, unsigned int seed);

// C++ interface function
void copy_stochastic_cuda(torch::Tensor target, torch::Tensor source) {
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(source.is_cuda(), "source must be a CUDA tensor");
    TORCH_CHECK(target.scalar_type() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(source.scalar_type() == torch::kFloat32, "source must be float32");
    TORCH_CHECK(target.numel() == source.numel(), "target and source must have the same number of elements");
    TORCH_CHECK(target.data_ptr() != source.data_ptr(), "target and source must not be the same tensor");

    // get a seed
    unsigned int seed = static_cast<unsigned int>(clock());

    copy_stochastic_kernel_launcher(target, source, seed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_stochastic_cuda", &copy_stochastic_cuda, "Copy stochastic (CUDA)");
}
