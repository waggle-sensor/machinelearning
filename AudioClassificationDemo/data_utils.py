"""
data_utils.py

Description: utilty functions used for audio data processing.
"""

##############################

import os
import tensorflow as tf
import numpy as np
from math import floor
from random import randint
from PIL import Image
import librosa

##############################

def downloadUrbanSound8k():
    """ Downloads UrbanSound8k dataset """
    os.system('wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz -O urban8k.tgz')
    os.system('tar -xzf urban8k.tgz')

##############################

def LogMelSpectMesh(y,sr=22050):
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]

    # Zero-padding for clip(size <= 2205)
    if len(y) <= 2205:
        clip = np.concatenate((y, np.zeros(2205 - len(y) + 1)))

    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i]*sr/1000))
        hop_length = int(round(hop_sizes[i]*sr/1000))
        spec = librosa.feature.melspectrogram(y,sr=22050,n_fft=2205, win_length=window_length,\
                                             hop_length=hop_length,n_mels=128)
        spec = librosa.power_to_db(spec)
        spec = np.asarray(Image.fromarray(spec).resize((250,128)))

        # Scale between [0,1]
        spec = (spec - np.min(spec))/np.ptp(spec)

        specs.append(spec)

    specs = np.array(specs)
    specs = np.moveaxis(specs, 0, 2)
    return specs

##############################

def MFCCMesh(y,sr=22050):
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]

    # Zero-padding for clip(size <= 2205)
    if len(y) <= 2205:
        clip = np.concatenate((y, np.zeros(2205 - len(y) + 1)))

    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i]*sr/1000))
        hop_length = int(round(hop_sizes[i]*sr/1000))
        spec = librosa.feature.mfcc(y,sr=22050,n_fft=2205, win_length=window_length,\
                                             hop_length=hop_length,n_mfcc=128)
        spec = librosa.power_to_db(spec)
        spec = np.asarray(Image.fromarray(spec).resize((250,128)))

        # Scale between [0,1]
        spec = (spec - np.min(spec))/np.ptp(spec)

        specs.append(spec)

    specs = np.array(specs)
    specs = np.moveaxis(specs, 0, 2)
    return specs

##############################

def ChromaMesh(y,sr=22050):
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]

    # Zero-padding for clip(size <= 2205)
    if len(y) <= 2205:
        clip = np.concatenate((y, np.zeros(2205 - len(y) + 1)))

    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i]*sr/1000))
        hop_length = int(round(hop_sizes[i]*sr/1000))
        spec = librosa.feature.chroma_stft(y,sr=22050,n_fft=2205, win_length=window_length,\
                                             hop_length=hop_length,n_chroma=128)
        spec = librosa.power_to_db(spec)
        spec = np.asarray(Image.fromarray(spec).resize((250,128)))

        # Scale between [0,1]
        spec = (spec - np.min(spec))/np.ptp(spec)

        specs.append(spec)

    specs = np.array(specs)
    specs = np.moveaxis(specs, 0, 2)
    return specs

##############################

def writeMelSpect(df):
    paths, labels = df.iloc[:,0].tolist(),df.iloc[:,2].tolist()
    name = paths[0].split("/")[2]
    record_name = 'UrbanSound8K/tfRecords/MelSpect/' + name + '.tfrec'

    with tf.io.TFRecordWriter(record_name) as writer:
        for i, audio_path in enumerate(paths):
            # Add sample to tfRecord file
            audio_data, _ = librosa.load(audio_path, sr=22050)
            spect = LogMelSpectMesh(audio_data)
            example = createTF_Example(spect, labels[i])
            writer.write(example.SerializeToString())

##############################

def writeMFCC(df):
    paths, labels = df.iloc[:,0].tolist(),df.iloc[:,2].tolist()
    name = paths[0].split("/")[2]
    record_name = 'UrbanSound8K/tfRecords/MFCC/' + name + '.tfrec'

    with tf.io.TFRecordWriter(record_name) as writer:
        for i, audio_path in enumerate(paths):
            # Add sample to tfRecord file
            audio_data, _ = librosa.load(audio_path, sr=22050)
            spect = MFCCMesh(audio_data)
            example = createTF_Example(spect, labels[i])
            writer.write(example.SerializeToString())

##############################

def writeChroma(df):
    paths, labels = df.iloc[:,0].tolist(),df.iloc[:,2].tolist()
    name = paths[0].split("/")[2]
    record_name = 'UrbanSound8K/tfRecords/Chroma/' + name + '.tfrec'

    with tf.io.TFRecordWriter(record_name) as writer:
        for i, audio_path in enumerate(paths):
            # Add sample to tfRecord file
            audio_data, _ = librosa.load(audio_path, sr=22050)
            spect = ChromaMesh(audio_data)
            example = createTF_Example(spect, labels[i])
            writer.write(example.SerializeToString())

##############################

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def createTF_Example(X,y):
    """Returns an tf.train.Example for data sample."""
    feature = {
        'rows': _int64_feature(X.shape[0]),
        'cols': _int64_feature(X.shape[1]),
        'depth': _int64_feature(X.shape[2]),
        'spec': _bytes_feature(X.tobytes()),
        'y': _int64_feature(y),
        }
    return tf.train.Example(features=tf.train.Features(feature=feature))
