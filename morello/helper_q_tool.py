import os
import pickle
import warnings
import matplotlib.pyplot as plt
import pennylane as qml
import qiskit
import tensorflow as tf
import librosa
import librosa.display
from pennylane import numpy as np
from pennylane.templates import RandomLayers
from scipy.io import wavfile
from tensorflow import keras
from tqdm import tqdm

# numbers of wires
N_WIRES = 4

# for running at QPU
NOISE_MODE = False

# Initialize device based on noise_mode
dev = qml.device('qiskit.aer', wires=N_WIRES) if NOISE_MODE else qml.device("default.qubit", wires=N_WIRES)

# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(1, N_WIRES))


@qml.qnode(dev)
def circuit(phi=None):
    """Quantum circuit for processing image data."""
    # Encoding of 4 classical input values
    for j in range(N_WIRES):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(N_WIRES)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(N_WIRES)]


def quantum_convolution(image, kernel_radius=2):
    """Applies the quantum circuit to an image."""
    h_feat, w_feat, ch_n = image.shape
    out = np.zeros((h_feat // kernel_radius, w_feat // kernel_radius, N_WIRES))

    for j in range(0, h_feat, kernel_radius):
        for k in range(0, w_feat, kernel_radius):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(
                phi=[image[j, k, 0], image[j, k + 1, 0], image[j + 1, k, 0], image[j + 1, k + 1, 0]]
            )
            for c in range(N_WIRES):
                out[j // kernel_radius, k // kernel_radius, c] = q_results[c]

    return out


def generate_quantum_speech(x_train, x_valid, kernel_radius):
    """Applies quantum convolution to a set of images."""
    q_train = [quantum_convolution(img, kernel_radius) for img in
               tqdm(x_train, desc="Quantum pre-processing of train Speech:")]
    q_valid = [quantum_convolution(img, kernel_radius) for img in
               tqdm(x_valid, desc="Quantum pre-processing of test Speech:")]

    return np.asarray(q_train), np.asarray(q_valid)


def plot_accuracy_loss(att_history, qatt_history, cnn_history, qcnn_history, data_ix):
    """Plots training accuracy and loss."""
    plt.figure()
    plt.style.use("seaborn")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

    ax1.plot(att_history.history["val_accuracy"], "-ok", label="Baseline Attn-BiLSTM")
    ax1.plot(qatt_history.history["val_accuracy"], "-ob", label="Attention With Quanv Layer")
    ax1.plot(cnn_history.history["val_accuracy"], "-ob", label="Baseline CNN")
    ax1.plot(qcnn_history.history["val_accuracy"], "-og", label="Quantum CNN")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(att_history.history["val_loss"], "-ok", label="Baseline Attn-BiLSTM")
    ax2.plot(qatt_history.history["val_loss"], "-og", label="Attention With Quanv Layer")
    ax2.plot(cnn_history.history["val_loss"], "-ob", label="Baseline CNN")
    ax2.plot(qcnn_history.history["val_loss"], "-og", label="Quantum CNN")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"images/{data_ix}_conv_speech_loss.png")


def display_speech(x_train, q_train, use_ch, temp_filename="tmp.png"):
    """Displays input speech and processed speech."""
    plt.figure()
    plt.subplot(5, 1, 1)

    if not use_ch:
        librosa.display.specshow(librosa.power_to_db(x_train[0, :, :, 0], ref=np.max))
    else:
        librosa.display.specshow(librosa.power_to_db(x_train[0, :, :], ref=np.max))

    plt.title('Input Speech')

    for i in range(4):
        plt.subplot(5, 1, i + 2)
        librosa.display.specshow(librosa.power_to_db(q_train[0, :, :, i], ref=np.max))
        plt.title(f'Channel {i + 1}: Quantum Compressed Speech')

    plt.tight_layout()
    plt.savefig(f"images/speech_encoder_{temp_filename}")

