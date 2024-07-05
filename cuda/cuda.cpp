#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include "AudioFile.h"
#include <chrono>
#include <utility>

__global__ void find_major_frequencies(cufftComplex *data, int block_size, int batch)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < (block_size / 2 + 1) * batch)
    {
        float magnitude = 2. / block_size * sqrt(data[idx].x * data[idx].x + data[idx].y * data[idx].y);
        float magnitude_db = 20 * log10f(magnitude);
        data[idx].x = magnitude_db;
        data[idx].y = 0;
    }
}

int main(int argc, char *argv[])
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    size_t block_size = 2048;
    size_t max_batch_size = 1000;

    std::string path = "../nicht_zu_laut_abspielen(1).wav";
    int step = 1, db_threshold = 50;
    float block_size_f = static_cast<float>(block_size);

    AudioFile<int> audioFile;
    audioFile.load(path);

    int sampleRate = audioFile.getSampleRate();
    int numSamples = audioFile.getNumSamplesPerChannel();
    std::vector<int> samples = audioFile.samples[0];

    size_t iterations = (numSamples - block_size) / max_batch_size;
    size_t last_batch_size = (numSamples - block_size) % max_batch_size;
    size_t batch = max_batch_size;
    cufftHandle plan;
    cufftReal *data;
    cufftComplex *result;
    cudaMalloc(&data, batch * block_size * sizeof(cufftReal));
    cudaMalloc(&result, batch * (block_size / 2 + 1) * sizeof(cufftComplex));

    cufftPlan1d(&plan, block_size, CUFFT_R2C, batch);

    std::vector<float> cx(batch * block_size);
    int thr_per_blk = block_size / 2 + 1;
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

    auto start = std::chrono::steady_clock::now();
    for (int k = 0; k < iterations; k++)
    {
        for (size_t j = 0; j < batch; j++)
        {
            for (size_t i = 0; i < block_size; i++)
            {
                cx[block_size * j + i] = static_cast<float>(samples[max_batch_size * k + i + j]);
            }
        }
        cudaMemcpy(data, cx.data(), batch * block_size * sizeof(cufftReal), cudaMemcpyHostToDevice);
        cufftExecR2C(plan, data, result);
        cudaDeviceSynchronize();

        find_major_frequencies<<<blk_in_grid, thr_per_blk>>>(result, block_size, batch);

        cudaDeviceSynchronize();
        std::vector<cufftComplex> y((block_size / 2 + 1) * batch);
        cudaMemcpy(y.data(), result, batch * (block_size / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

        for (int j = 0; j < batch; j++)
        {
            std::vector<std::pair<int, int>> results(0);
            for (int i = 0; i < block_size / 2 + 1; i++)
            {
                if (y[j * (block_size / 2 + 1) + i].x > db_threshold)
                {
                    results.push_back(std::make_pair(frequencies[i], static_cast<int>(y[j * (block_size / 2 + 1) + i].x)));
                }
            }
            all_results.push_back(results);
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "FFT took: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "[s]" << std::endl;
    start = std::chrono::steady_clock::now();

    std::ofstream outputFile;
    outputFile.open("cuda_stats.txt");
    for (int i = 0; i < all_results.size(); i++)
    {
        outputFile << i << ":\t";
        for (auto &pair : all_results[i])
        {
            outputFile << pair.first << ":" << pair.second << ",";
        }
        outputFile << "\n";
    }
    outputFile << std::endl;
    outputFile.close();
    end = std::chrono::steady_clock::now();
    std::cout << "Writing to file took: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "[s]" << std::endl;
    cufftDestroy(plan);

    cudaFree(result);
    cudaFree(data);
    return 0;
}