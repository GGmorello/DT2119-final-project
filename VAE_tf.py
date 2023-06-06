import os
import numpy as np
import librosa
import librosa.display
from dataProcess import getData,phenomenoes,sub_word_units,padding,getDataset
from matplotlib import pyplot as plt
import soundfile as sf
import tensorflow as tf
from keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.list_physical_devices('GPU')

def vae_loss(encoder_inputs, outputs, z_mean, z_log_var):
    reconstruction_loss = tf.keras.losses.mse(encoder_inputs, outputs)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    return reconstruction_loss + kl_loss
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim),
                                            mean=0.0, stddev=1.0)
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon


def display_spectrogram(original, reconstructed):
    #take just one sample from the batch
    original=original[0].squeeze().cpu().detach().numpy()
    audio_signal=librosa.feature.inverse.mfcc_to_audio(original)
    # Normalize the audio signal
    normalized_signal = audio_signal / np.max(np.abs(audio_signal))
    # Specify the output file path
    output_path = 'original.wav'
    # Choose the desired bit depth and specify the subtype accordingly
    bit_depth = 24
    subtype = f'PCM_{bit_depth}'
    # Save the audio signal as a WAV file with the specified bit depth
    sf.write(output_path, normalized_signal, 22050, subtype=subtype)
    reconstructed = reconstructed[0].squeeze().cpu().detach().numpy()
    audio_signal=librosa.feature.inverse.mfcc_to_audio(reconstructed)
    # Normalize the audio signal
    normalized_signal = audio_signal / np.max(np.abs(audio_signal))
    # Specify the output file path
    output_path = 'reconstructed.wav'
    # Choose the desired bit depth and specify the subtype accordingly
    bit_depth = 24
    subtype = f'PCM_{bit_depth}'
    # Save the audio signal as a WAV file with the specified bit depth
    sf.write(output_path, normalized_signal, 22050, subtype=subtype)


data=np.load('data.npz',allow_pickle=True)['data'][()]
batch_size = 8
dataset=getDataset(data)


input_shape = (13, 44, 1)  # Assuming grayscale input, add a channel dimension

# Define the number of latent dimensions
latent_dim = 32

# Define the encoder model
encoder_inputs = tf.keras.Input(shape=input_shape)
x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(encoder_inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
# Define the decoder model
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(512, activation='relu')(latent_inputs)
x = layers.Dense(64 * (input_shape[0] // 4) * (input_shape[1] // 4), activation='relu')(x)
x = layers.Reshape((input_shape[0] // 4, input_shape[1] // 4, 64))(x)
x = layers.Conv2DTranspose(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D(size=(2, 2))(x)
decoder_outputs = layers.Conv2DTranspose(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
# Define the decoder model
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name='decoder')
# Define the VAE model
outputs = decoder(encoder(encoder_inputs)[2])
vae = tf.keras.Model(encoder_inputs, outputs, name='vae')

vae.compile(optimizer='adam', loss=vae_loss)

print(vae.summary())


#from data.npz read the data in the format: word, sample, samplingrate, lmfcc
labels=["bed","cat","dog","five","happy","left","marvin","sheila","six","stop"]


