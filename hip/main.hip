#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <complex>
#include <cmath>
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "hip/hip_runtime.h"
#include "rocfft/rocfft.h"
#include "AudioFile.h"

__global__ void find_major_frequencies(float2 *data, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        float magnitude = 2. / N * sqrt(data[idx].x * data[idx].x + data[idx].y * data[idx].y);
        float magnitude_db = 20 * log10(magnitude);
        data[idx].x = magnitude_db;
        data[idx].y = 0;
    }
}

int main(int argc, char *argv[])
{
    rocfft_setup();
    std::cout << "HIP FFT" << std::endl;
    size_t block_size = 2048;
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
    size_t Nbytes = N * sizeof(float);

    // Create HIP device buffer
    float *x;
    hipMalloc(&x, Nbytes);

    // Initialize data
    std::vector<float> cx(2 * N);
    for (size_t i = 0; i < N; i++)
    {
        cx[i] = static_cast<float>(samples[i]);
    }

    //  Copy data to device
    hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);
    rocfft_plan plan = nullptr;
    size_t length = N;
    size_t batch = 1;
    rocfft_plan_create(&plan, rocfft_placement_inplace,
                       rocfft_transform_type_real_forward, rocfft_precision_single,
                       1, &length, batch, nullptr);

    // Check if the plan requires a work buffer
    size_t work_buf_size = 0;
    rocfft_plan_get_work_buffer_size(plan, &work_buf_size);
    void *work_buf = nullptr;
    rocfft_execution_info info = nullptr;
    if (work_buf_size)
    {
        rocfft_execution_info_create(&info);
        hipMalloc(&work_buf, work_buf_size);
        rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size);
    }

    // Execute plan
    rocfft_execute(plan, (void **)&x, nullptr, info);

    // Wait for execution to finish
    hipDeviceSynchronize();

    // Clean up work buffer
    if (work_buf_size)
    {
        hipFree(work_buf);
        rocfft_execution_info_destroy(info);
    }

    // Destroy plan
    rocfft_plan_destroy(plan);

    // Launch kernel
    hipLaunchKernelGGL(find_major_frequencies, dim3(N / 2), dim3(1), 0, 0, (float2 *)x, N / 2);

    // Copy result back to host
    std::vector<float2> y(N);
    hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);

    // Print results
    for (size_t i = 0; i < N / 2; i++)
    {
        std::cout << y[i].x << ", " << y[i].y << std::endl;
    }

    // Free device buffer
    hipFree(x);

    rocfft_cleanup();
    return 0;
}
