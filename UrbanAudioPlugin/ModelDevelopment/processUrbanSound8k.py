"""
processUrbanSound8k.py
Description: process UrbanSound8k data which
creates UrbanSound8k sampled at 22.5 kHz, MelSpectrograms
for UrbanSound8k sampled at 22.5 kHz, augmented MelSpectrograms
for UrbanSound8k sampled at 22.5 kHz, tfRecords for MelSpectrograms
for UrbanSound8k sampled at 22.5 kHz, and tfRecords
for augmented MelSpectrograms for UrbanSound8k sampled at 22.5 kHz.
"""

# Path to install UrbanSounds8k dataset
# wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz -O urban8k.tgz
# tar -xzf urban8k.tgz
# mv UrbanSound8K UrbanSound8k

##############################

from data_utils import *

import os
from glob import glob

import multiprocessing as mp
from itertools import repeat

from PIL import Image

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn import preprocessing

import matplotlib


##############################
# Declare paths for data
##############################

data_path = 'UrbanSound8k'
audio_path = 'audio'
image_path = 'audio_22_5_spect'
image_aug_path = 'audio_22_5_spect_aug'
meta_path = 'metadata'

##############################
# Declare paths for background sounds
##############################

bn_street_1 = 'background_noise/background_noise_150993__saphe__street-scene-1.wav'
bn_street_2 = 'background_noise/207208__jormarp__high-street-of-gandia-valencia-spain.wav'
bn_park = 'background_noise/268903__yonts__city-park-tel-aviv-israel.wav'
bn_street_3 ='background_noise/173955__saphe__street-scene-3.wav'
bn_paths = [bn_street_1,bn_street_2,bn_park,bn_street_3]

##############################
# Main logic
##############################

if __name__ == "__main__":
    ##############################
    print('Load wav file paths into a dict')
    ##############################

    audio_folders = glob(data_path+"/"+audio_path+"/f*")
    audio_files = {}
    for folder in sorted(audio_folders):
        audio_files[folder.split("/")[-1]] = glob(folder+"/*.wav")
    n_samples = sum([len(audio_files[k]) for k in audio_files.keys()])

    ##############################
    print('Resample wav files at 22.5 kHz')
    ##############################

    new_sample_rate_dir = data_path+'/'+'audio_22_5_kHz'
    if not os.path.isdir(new_sample_rate_dir):
        os.mkdir(new_sample_rate_dir)

    for key, files in enumerate(audio_files):
        [resampleWAV(file,new_sample_rate_dir) for file in audio_files[files]]

    ##############################
    print('Load background sounds')
    ##############################

    bn_sounds = []
    for bn_path in bn_paths:
        bn, _ = librosa.load(bn_path, sr=22500)
        bn_sounds.append(bn)

    ##############################
    print('Load 22.5 kHz wav file paths into a dict')
    ##############################

    audio_folders = glob(data_path+"/audio_22_5_kHz"+"/f*")
    audio_files = {}
    for folder in sorted(audio_folders):
        audio_files[folder.split("/")[-1]] = glob(folder+"/*.wav")
    n_samples = sum([len(audio_files[k]) for k in audio_files.keys()])

    ##############################
    print('Make MelSpectrogram dataset')
    ##############################

    if not os.path.isdir(data_path+'/audio_22_5_spect'):
        os.mkdir(data_path+'/audio_22_5_spect')

    pool = mp.Pool(processes=mp.cpu_count())
    pool_args = tuple(list(zip(audio_files.keys(), repeat(audio_files))))
    pool.starmap(makeSpectSet, pool_args)
    pool.close()
    pool.join()

    ##############################
    print('Make Augmented MelSpectrogram dataset')
    ##############################

    if not os.path.isdir(data_path+'/audio_22_5_spect_aug'):
        os.mkdir(data_path+'/audio_22_5_spect_aug')

    pool = mp.Pool(processes=mp.cpu_count())
    pool_args = tuple(list(zip(audio_files.keys(), repeat(audio_files),repeat(bn_sounds))))
    pool.starmap(makeAugmentSpectSet, pool_args)

    ##############################
    print('Make tfRecord dirs')
    ##############################

    if not os.path.isdir(data_path+'/tfRecords'):
        os.mkdir(data_path+'/tfRecords')

    if not os.path.isdir(data_path+'/tfRecords_aug'):
        os.mkdir(data_path+'/tfRecords_aug')

    ##############################
    print('Load metadata')
    ##############################

    meta_data = pd.read_csv(data_path+'/'+meta_path+'/UrbanSound8K.csv')
    meta_data['duration'] = meta_data['end']-meta_data['start']

    ##############################
    print('Make tfRecords for MelSpect')
    ##############################

    image_folders = glob(data_path+"/"+image_path+"/f*")
    image_files = {}
    for folder in sorted(image_folders):
        image_files[folder.split("/")[-1]] = glob(folder+"/*.png")
    n_samples = sum([len(image_files[k]) for k in image_files.keys()])

    all_data = []
    for k in image_files.keys():
        labels, classIds = [], []
        for sample_path in image_files[k]:
            sample_name = sample_path.split("/")[-1].split(".")[0]+".wav"
            row = meta_data[meta_data['slice_file_name'] == sample_name]
            sample_label =  row['class'].values[0]
            class_id = row['classID'].values[0]
            labels.append(sample_label)
            classIds.append(class_id)
        new_df = pd.DataFrame(list(zip(image_files[k],labels,classIds)))
        new_df.name = str(k)
        all_data.append(new_df)

    for df in all_data:
        name = df.name
        record_name = data_path+'/tfRecords/' + name + '.tfrec'
        paths, labels = df.iloc[:,0].tolist(),df.iloc[:,2].tolist()
        with tf.io.TFRecordWriter(record_name) as writer:
            for i, img_path in enumerate(paths):
                # Add sample to tfRecord file
                image = tf.io.decode_png(tf.io.read_file(img_path))
                example = create_example(image, labels[i])
                writer.write(example.SerializeToString())

    ##############################
    print('Make tfRecords for augmened MelSpect')
    ##############################

    image_folders = glob(data_path+"/"+image_aug_path+"/f*")
    image_files = {}
    for folder in sorted(image_folders):
        image_files[folder.split("/")[-1]] = glob(folder+"/*.png")
    n_samples = sum([len(image_files[k]) for k in image_files.keys()])

    all_data = []
    for k in image_files.keys():
        labels, classIds = [], []
        for sample_path in image_files[k]:
            sample_name = sample_path.split("/")[-1].split("^")[0]+".wav"
            row = meta_data[meta_data['slice_file_name'] == sample_name]
            sample_label =  row['class'].values[0]
            class_id = row['classID'].values[0]
            labels.append(sample_label)
            classIds.append(class_id)
        new_df = pd.DataFrame(list(zip(image_files[k],labels,classIds)))
        new_df.name = str(k)
        all_data.append(new_df)

    for df in all_data:
        name = df.name
        record_name = data_path+'/tfRecords_aug/' + name + '.tfrec'
        paths, labels = df.iloc[:,0].tolist(),df.iloc[:,2].tolist()
        with tf.io.TFRecordWriter(record_name) as writer:
            for i, img_path in enumerate(paths):
                # Add sample to tfRecord file
                image = tf.io.decode_png(tf.io.read_file(img_path))
                example = create_example(image, labels[i])
                writer.write(example.SerializeToString())
