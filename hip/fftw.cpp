#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <complex>
#include <cmath>
#include "fftw.h"
#include "AudioFile.h"
#include <chrono>
#include <utility>

int main(int argc, char *argv[])
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    size_t block_size = 2048;
    // for some reason, max_batch_size cannot be set higher, even though the GPU has enough memory
    // else, the values are not correct
    size_t max_batch_size = 1000;

    std::string path = "../nicht_zu_laut_abspielen(1).wav";
    int step = 1, db_threshold = 50;
    float block_size_f = static_cast<float>(block_size);

    AudioFile<int> audioFile;
    audioFile.load(path);

    int sampleRate = audioFile.getSampleRate();
    int numSamples = audioFile.getNumSamplesPerChannel();
    // only look at the first channel
    std::vector<int> samples = audioFile.samples[0];

    std::vector<int> frequencies(0);
    float startValue = 0;
    float endValue = sampleRate / 2.;
    int num = block_size / 2;
    float increment = (endValue - startValue) / (num - 1);
    for (int i = 0; i < num; i++)
    {
        frequencies.push_back(static_cast<int>(i * increment));
    }

    double *in = new double[block_size];
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (block_size / 2 + 1));
    fftw_plan p = fftw_plan_dft_r2c_1d(block_size, in, out, FFTW_ESTIMATE);

    for (int i = 0; i < numSamples - block_size; i += step)
    {
        for (int j = 0; j < block_size; j++)
        {
            in[j] = static_cast<double>(samples[i + j]);
        }

        fftw_execute(p);
        for (int j = 0; j < block_size / 2 + 1; j++)
        {
            double re = out[j][0];
            double im = out[j][1];
            double mag = sqrt(re * re + im * im);
            mag = 20 * log10(mag / block_size_f);
            if (mag > db_threshold)
            {
                std::cout << i << ":\t" << frequencies[j] << ":" << mag << "," << std::endl;
            }
        }
    }

    return 0;
}
