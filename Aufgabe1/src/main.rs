use argparse::{ArgumentParser, Store};
use hound;
use itertools_num::linspace;
use rustfft::{num_complex::Complex, FftPlanner};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Minimal example.
fn main() {
    let mut block_size: usize = 2048;
    let mut path = String::new();
    let mut step = 1;
    let mut db_threshold = 50;
    {
        let mut ap = ArgumentParser::new();
        ap.set_description("Reads a wav file and writes the major frequencies to a text file");
        ap.refer(&mut block_size).add_option(
            &["-b", "--block_size"],
            Store,
            "Block size for FFT. Default is 2048.",
        );
        ap.refer(&mut path)
            .add_option(&["-p", "--path"], Store, "Path to the wav file")
            .required();
        ap.refer(&mut step).add_option(
            &["-s", "--step"],
            Store,
            "Step size for the FFT. Default is 1.",
        );
        ap.refer(&mut db_threshold).add_option(
            &["-t", "--threshold"],
            Store,
            "Threshold for the magnitude of major frequencies in dB. Default is 50.",
        );
        ap.parse_args_or_exit();
    }
    let block_size_f = block_size as f32;

    // Get the path of the WAV file from the command line arguments

    // Open the WAV file
    let mut reader = hound::WavReader::open(&path).expect("Failed to open WAV file");

    // Create a vector to store the samples
    let samples = reader.samples::<i16>();
    let mut samples_vec: Vec<Complex<f32>> = samples
        .into_iter()
        .map(|sample| Complex {
            re: sample.unwrap() as f32,
            im: 0.0,
        })
        .collect();
    if reader.spec().channels == 2 {
        // If the WAV file has two channels, we only use the first channel
        samples_vec = samples_vec
            .into_iter()
            .step_by(2)
            .collect::<Vec<Complex<f32>>>();
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(block_size);
    let frequencies: Vec<f32> =
        linspace(0.0, reader.spec().sample_rate as f32 / 2.0, block_size / 2).collect();
    let start = Instant::now();
    println!(
        "Starting FFT computation on {} with blocksize {}, step {}...",
        path, block_size, step
    );
    let stats: Vec<(usize, Vec<(u32, u32)>)> = (0..samples_vec.len() - block_size)
        .step_by(step)
        .map(|i| {
            let buffer = &mut samples_vec[i..i + block_size].to_vec();
            fft.process(buffer);
            let mut block_stats = Vec::new();
            for j in 0..block_size / 2 {
                let freq = frequencies[j];
                let real = buffer[j].re;
                let imag = buffer[j].im;
                let magnitude = 2. / block_size_f * (real * real + imag * imag).sqrt();
                let magnitude_db = 20.0 * magnitude.log10();
                if magnitude_db > db_threshold as f32 {
                    block_stats.push((freq as u32, magnitude_db as u32));
                }
            }
            (i, block_stats)
        })
        .collect();
    let duration = start.elapsed();
    println!("FFT took: {:?}", duration);
    let start = Instant::now();
    // write stats to text file. Format should be: one line per block, startindex of block, all major frequencies
    let mut file = File::create("seq_stats.txt").expect("Failed to create file");
    for (idx, block) in stats.iter() {
        let mut line = format!("{}:\t", idx);
        for (freq, mag) in block.iter() {
            line.push_str(&format!("{}:{},", freq, mag));
        }
        line.push_str("\n");
        file.write_all(line.as_bytes())
            .expect("Failed to write to file");
    }
    file.flush().expect("Failed to flush file");
    let duration = start.elapsed();
    println!("Writing to file took: {:?}", duration);
}
