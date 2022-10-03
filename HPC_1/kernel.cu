#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error.h"
#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace std::chrono;

const char* FILE_PATH = "input.txt";
const int ASCII_SIZE = 256;

__global__ void charCalcKernel(char* text, int* charNums, int size)
{
    unsigned long int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < size)
    {
        atomicAdd(&charNums[(int)text[tid]], 1);
        tid += blockDim.x * gridDim.x;
    }
}

void subtractWithCUDA(char* text, int* charNums)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Create variables.
    int textLength = strlen(text);
    char* dev_text;
    int* dev_charNums;

    // Create time events.
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU buffers.
    HANDLE_ERROR(cudaMalloc((void**)&dev_text, textLength * sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_charNums, ASCII_SIZE * sizeof(int)));

    // Copy input data from host memory to GPU buffers.
    HANDLE_ERROR(cudaMemcpy(dev_text, text, textLength * sizeof(char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_charNums, charNums, ASCII_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    
    // Record start time.
    cudaEventRecord(start, nullptr);

    // Launch a kernel on the GPU.
    int threadPerBlock = prop.maxThreadsPerBlock;
    int blocksPerGrid = (textLength + threadPerBlock - 1) / threadPerBlock;
    printf_s("CUDA launch with %d blocks of %d threads.\n", blocksPerGrid, threadPerBlock);
    charCalcKernel<< <blocksPerGrid, threadPerBlock >> >(dev_text, dev_charNums, textLength);

    // cudaDeviceSynchronize waits for the kernel to finish, and return any errors encountered during the launch.
    HANDLE_ERROR(cudaDeviceSynchronize());
    
    //Calculate elapsed time.
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf_s(">Time on GPU = %.3f milliseconds.\n", time);

    // Copy output data from GPU buffer to host memory.
    HANDLE_ERROR(cudaMemcpy(charNums, dev_charNums, ASCII_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Free resources.
    cudaFree(dev_charNums);
    cudaFree(dev_text);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void charNumbersInStringWithCPU(char* text, int* charNums)
{
    auto start = high_resolution_clock::now();

    int textSize = strlen(text);
    for (int i = 0; i < textSize; ++i)
    {
        charNums[(int)text[i]]++;
    }

    auto end = high_resolution_clock::now();
    
    // Calculating total time taken by the program.
    double time_taken = duration_cast<nanoseconds>(end - start).count();
    printf_s(">Time on CPU = %.3f milliseconds.\n", time_taken *= 1e-6);
}

void readTextFromFile(const char* name, char** output)
{
    string text;
    ifstream in(name);
    if (in.is_open())
    {
        string line;
        while (getline(in, line))
        {
            text += ("\n" + line);
        }
    }
    else
    {
        throw invalid_argument("No such file!");
    }
    in.close();

    char* writable = new char[text.size() + 1];
    copy(text.begin(), text.end(), writable);
    writable[text.size()] = '\0';
    *output = writable;
}

bool isArraysEqual(const int* first, const int* second)
{
    int size = sizeof(first) / sizeof(int);

    for (int i = 0; i < size; ++i)
    {
        if (first[i] != second[i])
        {
            return false;
        }
    }
    return true;
}

void generateText(const int len, char** output)
{
    static const char alphanum[] =
        "0123456789 "
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    string tmp;
    tmp.reserve(len);

    for (int i = 0; i < len; ++i)
    {
        tmp += alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    char* writable = new char[tmp.size() + 1];
    copy(tmp.begin(), tmp.end(), writable);
    writable[tmp.size()] = '\0';
    *output = writable;
}

int main()
{
    srand(time(nullptr));
    char* text;
    double textSize = 0;
    // readTextFromFile(FILE_PATH, &text);
    
    printf("Data size: ");
    scanf("%lf", &textSize);
    generateText(textSize, &text);
    
    int numsGPU[ASCII_SIZE] = {0};
    subtractWithCUDA(text, numsGPU);
    int numsCPU[ASCII_SIZE] = {0};
    charNumbersInStringWithCPU(text, numsCPU);
    
    isArraysEqual(numsGPU, numsCPU)
        ? printf("\nGPU and CPU calculations equal.\n")
        : printf("\nGPU and CPU calculations NOT equal.\n");
    
    return 0;
}
