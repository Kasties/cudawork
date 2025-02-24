import torch
from torch.utils.cpp_extension import load_inline

# CUDA source code (cuda.cu)
cuda_kernel_source_grayscale = """
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(float *grayscale_image, const float *rgb_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int index = y * width + x;
        int rgb_index = index * 3;
        float r = rgb_image[rgb_index];
        float g = rgb_image[rgb_index + 1];
        float b = rgb_image[rgb_index + 2];
        grayscale_image[index] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

extern "C" void launch_rgb_to_grayscale_kernel(float *grayscale_image, const float *rgb_image, int width, int height) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayscale_image, rgb_image, width, height);
    cudaDeviceSynchronize();
}
"""

# C++ wrapper code (main.cpp) with a single module definition
cpp_wrapper_source_grayscale = """
#include <torch/extension.h>
#include <stdexcept>

// Declaration of helper functions from your CUDA source
extern "C" void launch_rgb_to_grayscale_kernel(float *grayscale_image, const float *rgb_image, int width, int height);

torch::Tensor rgb_to_grayscale_cuda(torch::Tensor rgb_tensor) {
    rgb_tensor = rgb_tensor.contiguous().cuda().to(torch::kFloat);
    if (rgb_tensor.dim() != 3 || rgb_tensor.size(2) != 3) {
        throw std::runtime_error("Input tensor must be a 3D RGB image (H x W x 3)");
    }
    int height = rgb_tensor.size(0);
    int width = rgb_tensor.size(1);

    auto grayscale_tensor = torch::empty({height, width}, torch::kFloat).cuda();

    launch_rgb_to_grayscale_kernel(
        grayscale_tensor.data_ptr<float>(),
        rgb_tensor.data_ptr<float>(),
        width,
        height
    );

    return grayscale_tensor.cpu();
}

// Only one module definition is allowed
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rgb_to_grayscale_cuda", &rgb_to_grayscale_cuda, "CUDA RGB to Grayscale conversion");
}
"""

try:
    # Compile the extension using load_inline
    grayscale_module = load_inline(
    name="grayscale_cuda_extension",
    cuda_sources=[cuda_kernel_source_grayscale],
    cpp_sources=[cpp_wrapper_source_grayscale],
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=["-arch=sm_75"]
)


    # Create a sample RGB image tensor (H x W x 3) on the GPU
    height, width = 256, 256
    rgb_image = torch.rand((height, width, 3), dtype=torch.float32, device="cuda")

    # Use the compiled CUDA function for grayscale conversion
    grayscale_cuda = grayscale_module.rgb_to_grayscale_cuda(rgb_image)

    # Compute a reference grayscale image using PyTorch's weighted sum:
    # (0.299 * R + 0.587 * G + 0.114 * B)
    grayscale_ref = (0.299 * rgb_image[..., 0] +
                    0.587 * rgb_image[..., 1] +
                    0.114 * rgb_image[..., 2]).cpu()

    # Compute the absolute difference between the two results
    difference = torch.abs(grayscale_cuda - grayscale_ref)
    max_diff = difference.max().item()

    print("Maximum difference between CUDA and reference conversion:", max_diff)

    if max_diff < 1e-5:
        print("Test Passed! The results match very closely.")
    else:
        print("Test Warning: There is a small numerical difference between the two methods.")

except Exception as e:
    print(f"Error during compilation or execution: {e}")
    print("Make sure you have CUDA and a compatible compiler installed.")
