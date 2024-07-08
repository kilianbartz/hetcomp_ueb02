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

void print_help()
{
    std::cout << "Usage: ./program_name [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
    std::cout << "  --path PATH            Path to the audio file (default: '../nicht_zu_laut_abspielen(1).wav')" << std::endl;
    std::cout << "  --block_size SIZE      Block size for FFT (default: 2048)" << std::endl;
    std::cout << "  --step STEP            Step size for sample processing (default: 1)" << std::endl;
    std::cout << "  --threshold THRESHOLD  Threshold in dB for frequency detection (default: 50)" << std::endl;
}

__global__ void find_major_frequencies(float2 *data, int block_size, int batch)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < (block_size / 2 + 1) * batch)
    {
        float magnitude = 2. / block_size * sqrt(data[idx].x * data[idx].x + data[idx].y * data[idx].y);
        float magnitude_db = 20 * log10(magnitude);
        data[idx].x = magnitude_db;
        data[idx].y = 0;
    }
}

int main(int argc, char *argv[])
{
    // see documentation, this is the maximum batch size that can be used to prevent errors
    const size_t max_batch_size = 1000;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // Default values
    std::string path = "../nicht_zu_laut_abspielen(1).wav";
    size_t block_size = 2048;
    int step = 1;
    int db_threshold = 50;

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0)
        {
            print_help();
            return 0;
        }

        if (i + 1 < argc)
        {
            if (strcmp(argv[i], "--path") == 0)
            {
                path = argv[i + 1];
                i++;
            }
            else if (strcmp(argv[i], "--block_size") == 0)
            {
                block_size = std::stoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "--step") == 0)
            {
                step = std::stoi(argv[i + 1]);
                i++;
            }
            else if (strcmp(argv[i], "--threshold") == 0)
            {
                db_threshold = std::stoi(argv[i + 1]);
                i++;
            }
            else
            {
                std::cerr << "Unknown option: " << argv[i] << std::endl;
                print_help();
                return 1;
            }
        }
    }

    // Print the values for confirmation
    std::cout << "Using the following parameters:" << std::endl;
    std::cout << "Path: " << path << std::endl;
    std::cout << "Block size: " << block_size << std::endl;
    std::cout << "Step: " << step << std::endl;
    std::cout << "Threshold: " << db_threshold << " dB" << std::endl;

    // read audio file
    AudioFile<int> audioFile;
    audioFile.load(path);

    int sampleRate = audioFile.getSampleRate();
    int numSamples = audioFile.getNumSamplesPerChannel();
    // only look at the first channel
    std::vector<int> samples = audioFile.samples[0];

    // configure hipfft for normal + last partial batch
    size_t iterations = (numSamples - block_size) / max_batch_size;
    size_t last_batch_size = (numSamples - block_size) % max_batch_size;
    size_t batch = max_batch_size;
    hipfftHandle plan;
    hipfftHandle lastPlan;
    hipfftReal *data;
    hipfftComplex *result;
    hipMalloc(&data, batch * block_size * sizeof(hipfftReal));
    hipMalloc(&result, batch * (block_size / 2 + 1) * sizeof(hipfftComplex));

    hipfftPlan1d(&plan, block_size, HIPFFT_R2C, batch);
    hipfftPlan1d(&lastPlan, block_size, HIPFFT_R2C, last_batch_size);

    int thr_per_blk = block_size / 2 + 1;
    int blk_in_grid = batch;

    std::vector<float> cx(batch * block_size);
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
    for (int k = 0; k <= iterations; k++)
    {
        cx.clear();
        size_t current_batch = (k < iterations) ? batch : last_batch_size;
        // init data
        for (size_t j = 0; j < current_batch; j++)
        {
            for (size_t i = 0; i < block_size; i++)
            {
                cx[block_size * j + i] = static_cast<float>(samples[max_batch_size * k + i + j]);
            }
        }
        hipMemcpy(data, cx.data(), current_batch * block_size * sizeof(hipfftReal), hipMemcpyHostToDevice);

        if (k < iterations)
        {
            hipfftExecR2C(plan, data, result);
            hipLaunchKernelGGL(find_major_frequencies, dim3(thr_per_blk), dim3(blk_in_grid), 0, 0, (float2 *)result, block_size, batch);
        }
        else
        {
            hipfftExecR2C(lastPlan, data, result);
            hipLaunchKernelGGL(find_major_frequencies, dim3(thr_per_blk), dim3(current_batch), 0, 0, (float2 *)result, block_size, batch);
        }
        hipDeviceSynchronize();
        std::vector<float2> y((block_size / 2 + 1) * batch);
        hipMemcpy(y.data(), result, current_batch * (block_size / 2 + 1) * sizeof(hipfftComplex), hipMemcpyDeviceToHost);

        for (int j = 0; j < current_batch; j++)
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

    // Print results
    std::ofstream outputFile;
    outputFile.open("hip_stats.txt");
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
    hipfftDestroy(plan);

    // Free device buffer
    hipFree(result);
    hipFree(data);
    return 0;
}
