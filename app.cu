#include <stdio.h>

__global__ void matrixmulkernel(float* M, float* N, float* P, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        float Pvalue = 0;
        for (int k = 0; k < width; k++) {
            Pvalue += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = Pvalue;
    }
}

int main() {
    int width = 3;
    float M[width][width], N[width][width], P[width][width];
    float *Md, *Nd, *Pd;
    int size = width * width * sizeof(float);

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            M[i][j] = 2;
            N[i][j] = 4;
        }
    }

    cudaMalloc((void**)&Md, size);
    cudaMalloc((void**)&Nd, size);
    cudaMalloc((void**)&Pd, size);

    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(width, width);
    dim3 dimGrid(1, 1);

    matrixmulkernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);

    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", P[i][j]);
        }
        printf("\n");
    }

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);

    return 0;
}