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
#include <utility>

__global__ void find_major_frequencies(float2 *data, int block_size, int batch)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < block_size / 2 * batch)
    {
        float magnitude = 2. / block_size * sqrt(data[idx].x * data[idx].x + data[idx].y * data[idx].y);
        float magnitude_db = 20 * log10(magnitude);
        data[idx].x = magnitude_db;
        data[idx].y = 0;
    }
}

int main(int argc, char *argv[])
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    size_t block_size = 2048;
    size_t max_batch_size = 2;

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

    size_t iterations = 1;
    // size_t iterations = floor((static_cast<float>(numSamples) - block_size) / step / max_batch_size);
    size_t last_batch_size = (numSamples - block_size) % max_batch_size;
    size_t batch = max_batch_size;
    hipfftHandle plan;
    hipfftReal *data;
    hipfftComplex *result;
    hipMalloc(&data, batch * block_size * sizeof(hipfftReal));
    hipMalloc(&result, batch * (block_size / 2 + 1) * sizeof(hipfftComplex));

    hipfftPlan1d(&plan, block_size, HIPFFT_R2C, batch);

    std::vector<float> cx(batch * block_size);
    int thr_per_blk = block_size / 2;
    int blk_in_grid = batch;
    std::vector<std::vector<std::pair<int, int>>> all_results(0);

    std::vector<int> frequencies(0);
    float startValue = 0;
    float endValue = sampleRate / 2.;
    int num = block_size / 2;
    float increment = (endValue - startValue) / (num - 1);
    for (int i = 0; i < num; i++)
    {
        frequencies.push_back(static_cast<int>(i * increment));
    }

    for (int k = 0; k < iterations; k++)
    {
        // init data
        for (size_t j = 0; j < batch; j++)
        {
            for (size_t i = 0; i < block_size; i++)
            {
                cx[block_size * j + i] = static_cast<float>(samples[max_batch_size * k + i + j]);
            }
        }
        hipMemcpy(data, cx.data(), batch * block_size * sizeof(hipfftReal), hipMemcpyHostToDevice);
        hipfftExecR2C(plan, data, result);
        hipLaunchKernelGGL(find_major_frequencies, dim3(thr_per_blk), dim3(blk_in_grid), 0, 0, (float2 *)result, block_size, batch);
        hipDeviceSynchronize();
        std::vector<float2> y((block_size / 2 + 1) * batch);
        hipMemcpy(y.data(), result, batch * (block_size / 2 + 1) * sizeof(hipfftComplex), hipMemcpyDeviceToHost);

        for (int j = 0; j < batch; j++)
        {

            std::vector<std::pair<int, int>> results(0);
            for (int i = 0; i < block_size / 2; i++)
            {
                if (y[j * (block_size / 2 + 1) + i].x > db_threshold)
                {
                    results.push_back(std::make_pair(frequencies[i], static_cast<int>(y[j * (block_size / 2 + 1) + i].x)));
                }
            }
            all_results.push_back(results);
        }
    }

    // Print results
    for (auto &val : all_results)
    {
        for (auto &pair : val)
        {
            std::cout << pair.first << ":" << pair.second << ", ";
        }
        std::cout << std::endl;
    }
    hipfftDestroy(plan);

    // Free device buffer
    hipFree(result);
    hipFree(data);
    return 0;
}
