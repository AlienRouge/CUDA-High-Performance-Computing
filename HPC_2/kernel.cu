#include "cuda_runtime.h"
#include "error.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace std::chrono;

const int BLOCK_SIZE = 32;

// https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/

__global__ void matrixMultKernel(const int* A, const int* B, int* C, int Acols, int Bcols)
{
    int thRow = blockIdx.y * blockDim.y + threadIdx.y;
    int thCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (thRow < Bcols && thCol < Bcols)
    {
        int i0 = Acols * (blockDim.y * blockIdx.y +
            threadIdx.y);
        int j0 = blockDim.x * blockIdx.x + threadIdx.x;

        int sum = 0;
        for (int k = 0; k < Acols; k++)
            sum += A[i0 + k] * B[k * Bcols + j0];
        int ind = Bcols * (blockDim.y * blockIdx.y +
            threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
        C[ind] = sum;
    }
}

__global__ void matrixMultKernelBase(int* A, int* B, int* C, int rows, int cols)
{
    int thRow = blockIdx.y * blockDim.y + threadIdx.y;
    int thCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (thRow < rows && thCol < rows)
    {
        int tmpSum = 0;
        for (int i = 0; i < cols; i++)
        {
            tmpSum += A[thRow * cols + i] * B[i * rows + thCol];
        }
        C[thRow * rows + thCol] = tmpSum;
    }
}

int* matrixMultiplyWithCPU(const int* A, const int* B, int rows, int cols)
{
    int* res = new int[rows * rows]{0};

    auto start = high_resolution_clock::now();

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < rows; ++j)
            for (int k = 0; k < cols; ++k)
                res[i * rows + j] += A[i * cols + k] * B[k * rows + j];

    // Calculating total time taken by the program.
    auto end = high_resolution_clock::now();
    double time_taken = duration_cast<nanoseconds>(end - start).count();
    printf_s(">Time on CPU = %.3f milliseconds.\n", time_taken *= 1e-6);

    return res;
}

int* matrixMultiplyWithGPU(const int* A, const int* B, int rows, int cols)
{
    // Create variables.
    int resultMatrixSize = rows * cols;
    int* res = new int[resultMatrixSize]{0};
    int* dev_A;
    int* dev_B;
    int* dev_C;

    // Create time events.
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU buffers.
    HANDLE_ERROR(cudaMalloc((void**)&dev_A, rows * cols * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_B, rows * cols * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_C, resultMatrixSize * sizeof(int)));

    // Copy input data from host memory to GPU buffers.
    HANDLE_ERROR(cudaMemcpy(dev_A, A, rows * cols * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_B, B, rows * cols * sizeof(int), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(rows, rows);
    dim3 blocksPerGrid(1, 1);
    if (rows * rows > BLOCK_SIZE * BLOCK_SIZE)
    {
        threadsPerBlock.x = BLOCK_SIZE;
        threadsPerBlock.y = BLOCK_SIZE;
        blocksPerGrid.x = (rows + threadsPerBlock.x - 1) / threadsPerBlock.x;
        blocksPerGrid.y = (rows + threadsPerBlock.y - 1) / threadsPerBlock.y;
    }
    printf_s("CUDA launch with %dx%d blocks of %dx%d threads.\n", blocksPerGrid.x, blocksPerGrid.y,
             threadsPerBlock.x,
             threadsPerBlock.y);

    // Record start time.
    cudaEventRecord(start, nullptr);

    // Launch a kernel on the GPU.
    // multiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C, rows, cols);
    matrixMultKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C, cols, rows);

    // cudaDeviceSynchronize waits for the kernel to finish, and return any errors encountered during the launch.
    HANDLE_ERROR(cudaDeviceSynchronize());

    //Calculate elapsed time.
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf_s(">Time on GPU = %.3f milliseconds.\n", time);

    // Copy output data from GPU buffer to host memory.
    HANDLE_ERROR(cudaMemcpy(res, dev_C, resultMatrixSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Free resources.
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return res;
}

int* generateIntMatrix(int rows, int cols)
{
    const int matrixSize = rows * cols;
    int* result = new int[matrixSize];

    for (int i = 0; i < matrixSize; ++i)
    {
        result[i] = rand() % 1000;
    }

    return result;
}

void showMatrix(const int* matrix, int mRows, int mCols)
{
    for (int i = 0; i < mRows; ++i)
    {
        for (int j = 0; j < mCols; ++j)
        {
            cout << matrix[i * mCols + j] << ' ';
        }
        cout << '\n';
    }
}

bool isMatrixEqual(int* A, int* B, int len)
{
    for (int i = 0; i < len; ++i)
    {
        if (A[i] != B[i])
        {
            return false;
        }
    }

    return true;
}

int main()
{
    srand(time(nullptr));
    int rows = 0;
    int cols = 0;

    printf_s("Enter the matrix size (x,y): ");
    cin >> rows >> cols;

    int* A = generateIntMatrix(rows, cols);
    int* B = generateIntMatrix(rows, cols);

    int* mGPU = matrixMultiplyWithGPU(A, B, rows, cols);
    int* mCPU = matrixMultiplyWithCPU(A, B, rows, cols);

    isMatrixEqual(mGPU, mCPU, rows * cols)
        ? printf("\nGPU and CPU calculations equal.\n")
        : printf("\nGPU and CPU calculations NOT equal.\n");

    return 0;
}
