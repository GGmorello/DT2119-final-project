import os
import argparse
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Local Definitions
from data_generator import generate_mel_spectrogram
from models import CNN_Model, Dense_Model, AttentionRNN_Model
from helper_q_tool import generate_quantum_speech, plot_accuracy_loss, display_speech

# Set the GPU to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Generate a unique timestamp for current run
current_time_stamp = time.strftime("%m%d_%H%M")

# Define labels
speech_commands = [
    'left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop',
]

# Paths
training_audio_path = 'dataset/'
data_save_path = "data_quantum/"  # Data saving folder

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="Epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
parser.add_argument("--sampling_rate", type=int, default=16000, help="Sampling Rate for input Speech")
parser.add_argument("--model_type", type=int, default=1, help="(0) Dense Model, (1) U-Net RNN Attention")
parser.add_argument("--mel_option", type=int, default=0, help="(0) Load Demo Features, (1) Generate Mel Features")
parser.add_argument("--quantum_option", type=int, default=1, help="(0) Load Demo Features, (1) Generate Quantum Features")
parser.add_argument("--partition_ratio", type=int, default=100, help="(1/N) data ratio for encoding ")
args = parser.parse_args()


def generate_training_data(labels, training_audio_path, sampling_rate, partition_ratio):
    all_wave, all_label = generate_mel_spectrogram(labels, training_audio_path, sampling_rate, partition_ratio)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(all_label)
    classes = list(label_encoder.classes_)
    y = keras.utils.to_categorical(y, num_classes=len(labels))

    x_train, x_valid, y_train, y_valid = train_test_split(
        np.array(all_wave), np.array(y), stratify=y, test_size=0.2, random_state=777, shuffle=True
    )
    height_feature, width_feature, _ = x_train[0].shape
    np.save(data_save_path + "n_x_train_speech.npy", x_train)
    np.save(data_save_path + "n_x_test_speech.npy", x_valid)
    np.save(data_save_path + "n_y_train_speech.npy", y_train)
    np.save(data_save_path + "n_y_test_speech.npy", y_valid)
    print("=== Feature Shape:", height_feature, width_feature)

    return x_train, x_valid, y_train, y_valid


def generate_quantum_features(x_train, x_valid, kernel_radius):
    print("Kernel Radius =", kernel_radius)
    q_train, q_valid = generate_quantum_speech(x_train, x_valid, kernel_radius)

    np.save(data_save_path + "demo_t1.npy", q_train)
    np.save(data_save_path + "demo_t2.npy", q_valid)

    return q_train, q_valid


if args.mel_option == 1:
    x_train, x_valid, y_train, y_valid = generate_training_data(speech_commands, training_audio_path, args.sampling_rate, args.partition_ratio)
else:
    x_train = np.load(data_save_path + "n_x_train_speech.npy")
    x_valid = np.load(data_save_path + "n_x_test_speech.npy")
    y_train = np.load(data_save_path + "n_y_train_speech.npy")
    y_valid = np.load(data_save_path + "n_y_test_speech.npy")

if args.quantum_option == 1:
    q_train, q_valid = generate_quantum_features(x_train, x_valid, 2)
else:
    q_train = np.load(data_save_path + "q_train_demo.npy")
    q_valid = np.load(data_save_path + "q_test_demo.npy")

# Early stopping & Model checkpointing
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
checkpoint = ModelCheckpoint('checkpoints/best_demo.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Model selection
if args.model_type == 0:
    model = Dense_Model(x_train[0], speech_commands)
elif args.model_type == 1:
    model = AttentionRNN_Model(q_train[0], speech_commands)

# Display model summary
model.summary()

# Model training
history = model.fit(x=q_train,
                    y=y_train,
                    epochs=args.epochs,
                    callbacks=[checkpoint],
                    batch_size=args.batch_size,
                    validation_data=(q_valid, y_valid))

# Save the model
model.save('checkpoints/' + current_time_stamp + '_demo.hdf5')

print("=== Batch Size:", args.batch_size)

