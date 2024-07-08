import argparse
import numpy as np
from scipy.io import wavfile
from scipy import signal


def generate_wave(freq, duration, wave_type="sine", sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    if wave_type == "sine":
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == "square":
        return signal.square(2 * np.pi * freq * t)
    elif wave_type == "sawtooth":
        return signal.sawtooth(2 * np.pi * freq * t)
    elif wave_type == "triangle":
        return signal.sawtooth(2 * np.pi * freq * t, width=0.5)
    else:
        raise ValueError(f"Unsupported wave type: {wave_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a WAV file with specified frequencies and wave types."
    )
    parser.add_argument("frequencies", type=float, nargs="+", help="Frequencies in Hz")
    parser.add_argument(
        "-d", "--duration", type=float, default=5.0, help="Duration in seconds"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output.wav", help="Output file path"
    )
    parser.add_argument(
        "-a",
        "--amplitude",
        type=float,
        default=0.5,
        help="Amplitude of the wave (0.0 to 1.0)",
    )
    parser.add_argument(
        "-s", "--sample-rate", type=int, default=44100, help="Sample rate in Hz"
    )
    parser.add_argument(
        "-w",
        "--wave-type",
        type=str,
        default="sine",
        choices=["sine", "square", "sawtooth", "triangle"],
        help="Type of wave to generate",
    )

    args = parser.parse_args()

    sample_rate = args.sample_rate
    duration = args.duration
    amplitude = args.amplitude
    wave_type = args.wave_type

    # Generate waves for each frequency and sum them
    wave = np.zeros(int(sample_rate * duration))
    for freq in args.frequencies:
        wave += generate_wave(freq, duration, wave_type, sample_rate)

    # Normalize and scale the wave
    wave = wave / len(args.frequencies)
    wave = np.clip(wave, -1, 1)  # Clip to prevent overflow
    wave = (wave * amplitude * 32767).astype(np.int16)

    # Write the WAV file
    wavfile.write(args.output, sample_rate, wave)
    print(f"WAV file generated: {args.output}")


if __name__ == "__main__":
    main()
