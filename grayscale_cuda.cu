
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(
    const float* input, float* output,
    int batch_size, int height, int width) {
    
    // Calculate global thread position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of pixels
    int total_pixels = batch_size * height * width;
    
    // Process pixels in grid-stride loop pattern
    for (int i = idx; i < total_pixels; i += blockDim.x * gridDim.x) {
        // Calculate position in each dimension
        int b = i / (height * width);
        int h = (i / width) % height;
        int w = i % width;
        
        // Calculate indices for RGB channels and grayscale output
        int r_idx = ((b * 3) + 0) * height * width + h * width + w;
        int g_idx = ((b * 3) + 1) * height * width + h * width + w;
        int b_idx = ((b * 3) + 2) * height * width + h * width + w;
        
        // Output index
        int out_idx = b * height * width + h * width + w;
        
        // Standard RGB to grayscale conversion
        output[out_idx] = 0.299f * input[r_idx] + 0.587f * input[g_idx] + 0.114f * input[b_idx];
    }
}

__global__ void rgb_to_grayscale_nchw_kernel(
    const float* input, float* output,
    int batch_size, int height, int width) {
    
    // Calculate global thread position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of pixels
    int total_pixels = batch_size * height * width;
    
    // Process pixels in grid-stride loop pattern
    for (int i = idx; i < total_pixels; i += blockDim.x * gridDim.x) {
        // Calculate position in each dimension
        int b = i / (height * width);
        int h = (i / width) % height;
        int w = i % width;
        
        // Input has shape (batch_size, 3, height, width)
        int r_idx = (b * 3 + 0) * height * width + h * width + w;
        int g_idx = (b * 3 + 1) * height * width + h * width + w;
        int b_idx = (b * 3 + 2) * height * width + h * width + w;
        
        // Output has shape (batch_size, 1, height, width)
        int out_idx = b * height * width + h * width + w;
        
        // Standard RGB to grayscale conversion
        output[out_idx] = 0.299f * input[r_idx] + 0.587f * input[g_idx] + 0.114f * input[b_idx];
    }
}

// C++ wrapper for the CUDA kernel
void rgb_to_grayscale_cuda(
    torch::Tensor input,
    torch::Tensor output) {
    
    // Get dimensions
    int batch_size = input.size(0);
    int height = input.size(2);
    int width = input.size(3);
    
    // Get pointers to data
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Configure grid and blocks
    int threads = 256;
    int blocks = (batch_size * height * width + threads - 1) / threads;
    
    // Launch the kernel
    rgb_to_grayscale_nchw_kernel<<<blocks, threads>>>(
        input_ptr, output_ptr, batch_size, height, width);
}
// Function declarations for Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rgb_to_grayscale", &rgb_to_grayscale_cuda, "RGB to Grayscale conversion (CUDA)");
}
        