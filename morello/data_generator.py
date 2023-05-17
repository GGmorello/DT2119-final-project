import os
import librosa
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Define the path to the training audio data
TRAIN_AUDIO_PATH = 'dataset/'
SAMPLE_RATE = 16000


def generate_mel_spectrogram(labels, audio_path=TRAIN_AUDIO_PATH, sample_rate=SAMPLE_RATE, port=1):
    """
    Generate Mel spectrograms for audio data.

    Args:
    labels (list): Labels corresponding to the audio files.
    audio_path (str, optional): Path to the audio files. Defaults to TRAIN_AUDIO_PATH.
    sample_rate (int, optional): Sample rate to be used by librosa. Defaults to SAMPLE_RATE.
    port (int, optional): Fraction of samples to be taken. Defaults to 1.

    Returns:
    all_wave (list): Mel spectrograms for each audio file.
    all_label (list): Corresponding labels for each audio file.
    """

    all_wave = []
    all_label = []

    for label in tqdm(labels):
        # Get all .wav files for the current label
        wav_files = [f for f in os.listdir(os.path.join(audio_path, label)) if f.endswith('.wav')]

        for idx, wav_file in enumerate(wav_files):
            # Load the audio file
            audio, _ = librosa.load(os.path.join(audio_path, label, wav_file), sr=sample_rate)

            # If idx is a multiple of port, process the audio file
            if idx % port == 0:
                if len(audio) == sample_rate:
                    # Generate the Mel spectrogram
                    mel_spectrogram = librosa.feature.melspectrogram(
                        y=audio, sr=sample_rate, n_fft=1024, hop_length=128, power=1.0, n_mels=60, fmin=40.0,
                        fmax=sample_rate / 2
                    )

                    # Add a new axis to the Mel spectrogram and append it to all_wave
                    all_wave.append(np.expand_dims(mel_spectrogram, axis=2))

                    # Append the label to all_label
                    all_label.append(label)

    return all_wave, all_label
