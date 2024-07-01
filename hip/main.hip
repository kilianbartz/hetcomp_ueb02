#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <complex>
#include <cmath>
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "hip/hip_runtime.h"
#include "hipfft/hipfft.h"
#include "AudioFile.h"
#include <chrono>

__global__ void find_major_frequencies(float2 *data, int N, int batch)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N / 2 * batch)
    {
        float magnitude = 2. / N * sqrt(data[idx].x * data[idx].x + data[idx].y * data[idx].y);
        float magnitude_db = 20 * log10(magnitude);
        data[idx].x = magnitude_db;
        data[idx].y = 0;
    }
}

int main(int argc, char *argv[])
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    size_t block_size = 2048;
    size_t batch = 2;

    std::string path = "../nicht_zu_laut_abspielen(1).wav";
    int step = 1, db_threshold = 50;
    std::cout << "Block size: " << block_size << std::endl;
    float block_size_f = static_cast<float>(block_size);

    AudioFile<int> audioFile;
    audioFile.load(path);

    int sampleRate = audioFile.getSampleRate();
    int numSamples = audioFile.getNumSamplesPerChannel();
    // only look at the first channel
    std::vector<int> samples = audioFile.samples[0];

    size_t N = 2048;

    // Create HIP device buffer

    // Initialize data
    std::vector<float> cx(batch * N);
    for (size_t j = 0; j < batch; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            cx[N * j + i] = static_cast<float>(samples[i + j]);
        }
    }
    hipfftHandle plan;
    hipfftReal *data;
    hipfftComplex *result;
    hipMalloc(&data, batch * block_size * sizeof(hipfftReal));
    hipMalloc(&result, batch * block_size * sizeof(hipfftComplex));
    hipMemcpy(data, cx.data(), batch * block_size * sizeof(hipfftReal), hipMemcpyHostToDevice);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Daten vorbereitet: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    //  Copy data to device

    hipfftPlan1d(&plan, block_size, HIPFFT_R2C, batch);

    hipfftExecR2C(plan, data, result);

    // Wait for execution to finish
    hipDeviceSynchronize();
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::cout << "FFT berechnet: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    // Launch kernel
    int thr_per_blk = N / 2;
    int blk_in_grid = batch;
    hipLaunchKernelGGL(find_major_frequencies, dim3(thr_per_blk), dim3(blk_in_grid), 0, 0, (float2 *)result, N, batch);

    // Copy result back to host
    std::vector<float2> y((N / 2 + 1) * batch);
    hipMemcpy(y.data(), result, batch * (block_size / 2 + 1) * sizeof(hipfftComplex), hipMemcpyDeviceToHost);

    // Print results
    for (float2 val : y)
    {
        std::cout << val.x << " " << val.y << std::endl;
    }
    hipfftDestroy(plan);

    // Free device buffer
    hipFree(result);
    hipFree(data);
    return 0;
}
