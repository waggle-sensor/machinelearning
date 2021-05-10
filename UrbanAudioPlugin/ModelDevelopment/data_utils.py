"""
data_utils.py

Description: utilty functions used for audio data processing.
"""

##############################

import os

import tensorflow as tf

import torch
import torchaudio
import torchvision

import numpy as np
from random import randint

from PIL import Image

import librosa
import librosa.display
import soundfile as sf

import matplotlib

##############################

def downloadUrbanSound8k():
    """ Downloads UrbanSound8k dataset """
    os.system('wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz -O urban8k.tgz')
    os.system('tar -xzf urban8k.tgz')
    os.system('mv UrbanSound8K UrbanSound8k')

##############################

def resampleWAV(wav_path,new_dir,new_sr=22500):
    """ Resamples wav file

    Parameters
    ----------
    wav_path : str
        The file location of wav file
    new_dir : str
        The dir location to store new wav file
    new_sr : int
        New sample rate

    """
    fold_name, wav_name = wav_path.split('/')[-2:]
    new_wav_name = new_dir+"/"+fold_name+"/"+wav_name
    if not os.path.isdir(new_dir+"/"+fold_name):
        os.mkdir(new_dir+"/"+fold_name)
    y, sr = librosa.load(wav_path, sr=new_sr)
    sf.write(new_wav_name, y, sr)

##############################

def add_background_noise(y,bn):
    """ Adds background noise to sound

    Parameters
    ----------
    y : np.array
        audio sample
    bn : np.array
        audio sample of background noise

    Returns
    -------
    np.array
        y sound overlapped with sample from background noies bn
    """
    # Sample uniform for loudness of noise
    w = np.random.uniform(.1,.5)
    # Get length of noise and audio
    bn_len = bn.shape[0]
    y_len = y.shape[0]
    # Get window of noise for bn
    bn_start = randint(0,bn_len-y_len)
    bn_end = bn_start+y_len
    bn_split = bn[bn_start:bn_end]
    # Add noise
    z = (1-w)*y+w*bn_split
    return z

##############################

def add_pitch_shift(y,step,sr=22500):
    """ Adds pitch shift to audio

    Parameters
    ----------
    y : np.array
        audio sample
    step : float
        says how much pitch shift to apply
    sr : int
        sample rate of audio

    Returns
    -------
    np.array
        audio with applied pitch shift
    """
    y_pitch_shift = librosa.effects.pitch_shift(y, sr, n_steps=step)
    return y_pitch_shift

##############################

def add_time_stretch(y,rate):
    """ Adds pitch shift to audio

    Parameters
    ----------
    y : np.array
        audio sample
    rate : float
        how much to speed up or slow down audio

    Returns
    -------
    np.array
        audio with applied time shift
    """
    y_stretch = librosa.effects.time_stretch(y, rate)
    return y_stretch

##############################

def make_MelSpectrogram(clip,sr=22050):
    """ Make MelSpectrogram from audio clip

    Parameters
    ----------
    clip : np.array
        audio sample
    sr : float
        sample rate of audio

    Returns
    -------
    np.array
        MelSpectrograms of audio clip
    """

    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]

    # Zero-padding for clip(size <= 2205)
    if len(clip) <= 2205:
        clip = np.concatenate((clip, np.zeros(2205 - len(clip) + 1)))

    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i]*sr/1000))
        hop_length = int(round(hop_sizes[i]*sr/1000))
        clip = torch.Tensor(clip)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2205, win_length=window_length, hop_length=hop_length, n_mels=128)(clip)
        eps = 1e-6
        spec = spec.numpy()
        spec = np.log(spec+ eps)
        spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
        # Scale between [0,1]
        spec = (spec - np.min(spec))/np.ptp(spec)
        specs.append(spec)
    specs = np.array(specs)
    specs = np.moveaxis(specs, 0, 2)
    return specs

##############################

def makeSpectSet(k,audio_files,data_path='UrbanSound8k'):
    """ Makes  MelSpectrogram dataset

    Parameters
    ----------
    k : int
        refers to fold of data
    audio_files(global) : dict
        container of audio paths
    data_path : str
        path of dataset
    """
    if not os.path.isdir(data_path+'/audio_22_5_spect/'+k):
        os.mkdir(data_path+'/audio_22_5_spect/'+k)
    for wav_file in audio_files[k]:
        file_name = wav_file.split("/")[-1].split(".")[0]
        new_path = data_path+'/audio_22_5_spect/'+k+"/"+file_name+".png"
        data, fs = sf.read(wav_file, dtype='float32')
        data_spect = make_MelSpectrogram(data)
        matplotlib.image.imsave(new_path, data_spect)


##############################

def makeAugmentSpectSet(k,audio_files,bn_sounds,data_path='UrbanSound8k'):
    """ Makes data augmented MelSpectrogram dataset

    Parameters
    ----------
    k : int
        refers to fold of data
    audio_files(globl) : dict
        container of audio paths
    data_path : str
        path of dataset
    """

    if not os.path.isdir(data_path+'/audio_22_5_spect_aug/'+k):
        os.mkdir(data_path+'/audio_22_5_spect_aug/'+k)
    for wav_file in audio_files[k]:
        file_name = wav_file.split("/")[-1].split(".")[0]
        new_path = data_path+'/audio_22_5_spect_aug/'+k+"/"+file_name
        data, fs = sf.read(wav_file, dtype='float32')

        # Add background noises 1
        new_data = add_background_noise(data,bn_sounds[0])
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^bn1.png", data_spect)

        # Add background noises 2
        new_data = add_background_noise(data,bn_sounds[1])
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^bn2.png", data_spect)

        # Add background noises 3
        new_data = add_background_noise(data,bn_sounds[2])
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^bn3.png", data_spect)

        # Add background noises 4
        new_data = add_background_noise(data,bn_sounds[3])
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^bn4.png", data_spect)

        # Add pitch shift -3.5
        new_data = add_pitch_shift(data,-3.5)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ps_n3h.png", data_spect)

        # Add pitch shift -2.5
        new_data = add_pitch_shift(data,-2.5)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ps_n2h.png", data_spect)

        # Add pitch shift -2
        new_data = add_pitch_shift(data,-2)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ps_n2.png", data_spect)

        # Add pitch shift -1
        new_data = add_pitch_shift(data,-1)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ps_n1.png", data_spect)

        # Add pitch shift 1
        new_data = add_pitch_shift(data,1)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ps_1.png", data_spect)

        # Add pitch shift 2
        new_data = add_pitch_shift(data,2)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ps_2.png", data_spect)

        # Add pitch shift 2.5
        new_data = add_pitch_shift(data,2.5)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ps_2h.png", data_spect)

        # Add pitch shift 3.5
        new_data = add_pitch_shift(data,3.5)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ps_3h.png", data_spect)

        # Add time shift .81
        new_data = add_time_stretch(data,.81)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ts_81.png", data_spect)

        # Add time shift .93
        new_data = add_time_stretch(data,.93)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ts_93.png", data_spect)

        # Add time shift 1.07
        new_data = add_time_stretch(data,1.07)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ts_107.png", data_spect)

        # Add time shift 1.23
        new_data = add_time_stretch(data,1.23)
        data_spect = make_MelSpectrogram(new_data)
        matplotlib.image.imsave(new_path+"^ts_123.png", data_spect)

##############################

def image_feature_png(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])
    )

##############################

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

##############################

def create_example(image,target):
    """Returns an tf.train.Example for data sample."""
    feature = {
        "image": image_feature_png(image),
        "target": int64_feature(target),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
