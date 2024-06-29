import numpy as np
import pyfftw
from scipy.io import wavfile
from multiprocessing import Pool, cpu_count
import pandas as pd
import psutil
import os
import threading
import time
import csv
from tqdm import tqdm
import gc

BLOCKSIZE = 2205  # 0.05 seconds
MEMORY_SAMPLING_INTERVAL = 3
DB_THRESHOLD = 50
NUM_RUNS = 1

pyfftw.interfaces.cache.enable()


def analyze_audio_block(blocks):
    # Get the current block
    stat_list = []
    for block in tqdm(blocks):
        y_block, idx, sr = block

        # if len(y_block) < BLOCKSIZE:
        #     # Letzter Block könnte kürzer als die Blockgröße sein
        #     return None

        # Berechnung der Fourier-Transformierten
        N = len(y_block)
        yf = pyfftw.interfaces.numpy_fft.fft(y_block, planner_effort="FFTW_ESTIMATE")
        xf = np.linspace(0.0, sr / 2.0, N // 2)
        magnitude = 2.0 / N * np.abs(yf[: N // 2])
        magnitude_db = 20 * np.log10(magnitude)
        max_magnitudes = np.sort(magnitude_db)[::-1]
        max_indices = np.argsort(magnitude_db)[::-1]
        # to_index should be the first index where the magnitude is smaller than the threshold
        to_index = np.where(max_magnitudes < DB_THRESHOLD)[0][0]
        major_frequencies = [
            # floor the frequency to the nearest integer
            (int(xf[max_indices[i]]), int(magnitude_db[max_indices[i]]))
            for i in range(to_index)
        ]

        # Speichern der Statistiken
        stats = {
            "block_start": idx,
            "block_end": idx + BLOCKSIZE,
            "major_frequencies": major_frequencies,
        }
        stat_list.append(stats)
    return stat_list


def analyze_audio_blocks(audio_file):
    # initialize csv header
    with open("statistics.csv", mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "block_start",
                "block_end",
                "major_frequencies",
            ]
        )

    # Laden der Audiodatei
    sr, y = wavfile.read(audio_file)
    y = y[:, 0]  # Convert to mono if stereo

    num_cpu = 1
    slice_size = len(y) // num_cpu

    # Prepare the blocks to be analyzed
    blocks = [
        (y[i * slice_size : (i + 1) * slice_size], i * slice_size, sr)
        for i in range(num_cpu)
    ]

    with Pool(processes=num_cpu) as pool:
        results = pool.map(process_chunk, blocks)

    # Collect and write the results
    for result in results:
        if result is not None:
            df = pd.DataFrame(result)
            df.to_csv("statistics.csv", mode="a", header=False, index=False)


def process_chunk(block):
    y_chunk, start_idx, sr = block
    stats_list = []
    blocks = [
        (y_chunk[i : i + BLOCKSIZE], start_idx + i, sr)
        for i in range(0, len(y_chunk) - BLOCKSIZE)
    ]
    stats_list = analyze_audio_block(blocks)
    return stats_list


if __name__ == "__main__":
    try:
        for i in range(NUM_RUNS):
            stop_mem_recording = False
            audio_file = "/home/kilian/hetcomp/Aufgabe2/nicht_zu_laut_abspielen.wav"

            analyze_audio_blocks(audio_file)
    except KeyboardInterrupt:
        print("Memory recording stopped")
        print("Exiting...")
        exit(0)
