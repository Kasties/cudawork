#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <chrono>
using namespace std::chrono;


int main() {
    cudaDeviceProp devProp;
    int devCount;

    // Get the number of CUDA devices
    cudaError_t error_id = cudaGetDeviceCount(&devCount);
    if (error_id != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceCount returned " << error_id << std::endl;
        std::cerr << "  " << cudaGetErrorString(error_id) << std::endl;
        return 1; // Indicate failure
    }

    // Check if any devices were found
    if (devCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 0; // No error, just no devices
    }

    std::cout << "Number of CUDA devices: " << devCount << std::endl;

    // Get properties for each device
    for (int i = 0; i < devCount; ++i) {
        error_id = cudaGetDeviceProperties(&devProp, i);
        if (error_id != cudaSuccess) {
            std::cerr << "Error: cudaGetDeviceProperties (device " << i << ") returned " << error_id << std::endl;
            std::cerr << "  " << cudaGetErrorString(error_id) << std::endl;
            continue; // Try the next device, or return 1; if you want to stop at the first error
        }

        std::cout << "\nDevice " << i << ": " << devProp.name << std::endl;
        std::cout << "  Compute Capability: " << devProp.major << "." << devProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << devProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl; // Convert to MB
        std::cout << "  Multiprocessors: " << devProp.multiProcessorCount << std::endl;
        std::cout << "  Warp Size: " << devProp.warpSize << std::endl;
        std::cout << "  Max Threads Per Block: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Clock Rate: " << devProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Clock Rate: " << devProp.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Total Constant Memory: " << devProp.totalConstMem / 1024 << " KB" << std::endl; // Convert to KB
        std::cout << "  Shared Memory Per Block: " << devProp.sharedMemPerBlock / 1024 << " KB" << std::endl; //Convert to KB
    }

    return 0; // Indicate success
}