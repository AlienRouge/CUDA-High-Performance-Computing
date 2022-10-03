#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

using namespace std::chrono;

__global__ void subtractKernel(int* a, int* b, int* c, int size)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < size)
    {
        c[tid] = a[tid] - b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int* initializeIntVector(const unsigned int size)
{
    int* vector = new int[size];
    for (int i = 0; i < size; i++)
    {
        vector[i] = (rand() % 100);
    }

    return vector;
}

bool isVectorsEqual(const int* first, const int* second, int size)
{
    for (int i = 0; i < size; ++i)
    {
        if (first[i] != second[i])
        {
            return false;
        }
    }
    return true;
}

void subtractWithCUDA(const int* a, const int* b, int* c, unsigned int size)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Create dev variables.
    int* dev_a;
    int* dev_b;
    int* dev_c;

    // Create time events.
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU buffers for three vectors (two input, one output).
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, size * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, size * sizeof(int)));

    // Copy input vectors from host memory to GPU buffers.
    HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

    // Record start time.
    cudaEventRecord(start, nullptr);

    // Launch a kernel on the GPU with XXX thread.
    int threadPerBlock = prop.maxThreadsPerBlock;
    int blocksPerGrid = (size + threadPerBlock - 1) / threadPerBlock;
    printf_s("CUDA launch with %d blocks and %d threads.\n", blocksPerGrid, threadPerBlock);
    subtractKernel << <blocksPerGrid, threadPerBlock >> >(dev_a, dev_b, dev_c, size);

    // cudaDeviceSynchronize waits for the kernel to finish, and return any errors encountered during the launch.
    HANDLE_ERROR(cudaDeviceSynchronize());

    //Calculate elapsed time.
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf_s(">Time on GPU = %.3f milliseconds.\n", time);

    // Copy output vector from GPU buffer to host memory.
    HANDLE_ERROR(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Free resources.
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void subtractWithCPU(const int* a, const int* b, int* c, unsigned int size)
{
    auto start = high_resolution_clock::now();

    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] - b[i];
    }

    auto end = high_resolution_clock::now();

    // Calculating total time taken by the program.
    double time_taken = duration_cast<nanoseconds>(end - start).count();
    printf_s(">Time on CPU = %.3f milliseconds.\n", time_taken *= 1e-6);
}

int main()
{
    srand(time(nullptr));
    int vectorSize = 0;

    printf_s("Input vectors size: ");
    scanf_s("%d", &vectorSize);

    const int* a = initializeIntVector(vectorSize);
    const int* b = initializeIntVector(vectorSize);
    int* cGPU = new int[vectorSize];
    int* cCPU = new int[vectorSize];

    subtractWithCUDA(a, b, cGPU, vectorSize);
    subtractWithCPU(a, b, cCPU, vectorSize);

    isVectorsEqual(cCPU, cGPU, vectorSize)
        ? printf("GPU and CPU calculations equal.\n")
        : printf("GPU and CPU calculations not equal.\n");

    return 0;
}
