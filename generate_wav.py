import numpy as np
import scipy.io.wavfile as wav
import argparse


def db_to_amplitude(db):
    """Konvertiert Dezibel (dB) in lineare Amplitude."""
    return 10 ** (db / 20)


def generate_sine_wave(frequency, duration, sample_rate, amplitude):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


def generate_square_wave(frequency, duration, sample_rate, amplitude):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    return wave


def generate_triangle_wave(frequency, duration, sample_rate, amplitude):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * 2 * (2 * ((t * frequency) % 1) - 1)
    wave = np.abs(wave) * 2 - 1
    return wave * amplitude


def generate_sawtooth_wave(frequency, duration, sample_rate, amplitude):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
    return wave


def save_wave(filename, data, sample_rate):
    wav.write(filename, sample_rate, data.astype(np.int16))


def main():
    parser = argparse.ArgumentParser(
        description="Generiert WAV-Dateien mit verschiedenen Wellentypen und Frequenzen."
    )
    parser.add_argument(
        "wave_type",
        choices=["sine", "square", "triangle", "sawtooth"],
        help="Der Typ der zu generierenden Welle",
    )
    parser.add_argument(
        "frequencies", type=float, nargs="+", help="Die Frequenzen der Wellen in Hz"
    )
    parser.add_argument(
        "--filename", type=str, help="Der Name der resultierenden WAV-Datei"
    )
    parser.add_argument(
        "--duration", type=float, default=2, help="Die Dauer der Audiodatei in Sekunden"
    )
    args = parser.parse_args()

    sample_rate = 44100
    duration = args.duration
    amplitude_db = 50
    amplitude = db_to_amplitude(amplitude_db)

    wave_generators = {
        "sine": generate_sine_wave,
        "square": generate_square_wave,
        "triangle": generate_triangle_wave,
        "sawtooth": generate_sawtooth_wave,
    }

    wave_type = args.wave_type
    frequencies = args.frequencies
    filename = args.filename

    combined_wave = np.zeros(int(sample_rate * duration))

    for frequency in frequencies:
        wave = wave_generators[wave_type](frequency, duration, sample_rate, amplitude)
        combined_wave += wave

    combined_wave = combined_wave / len(
        frequencies
    )  # Vermeidung von Übersteuerung durch Mittelung

    save_wave(filename, combined_wave, sample_rate)


if __name__ == "__main__":
    main()
