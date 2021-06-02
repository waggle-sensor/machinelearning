"""
train_model.py

Description: code used to train neural network over UrbanSound8k processed data.

Takes three arguments: --epochs (int > 0)
Ex: python train_model.py --epochs 5
"""

#######################################

# Import modules
from data_utils import *

import copy
import argparse
import os
from glob import glob

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers

from sklearn.metrics import accuracy_score

#######################################

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int)

#######################################

# Declare path to data
data_path_mel = "UrbanSound8K/tfRecords/MelSpect/fold"
data_path_chroma = "UrbanSound8K/tfRecords/Chroma/fold"
data_path_mfcc = "UrbanSound8K/tfRecords/MFCC/fold"

# Declare global variables
BATCH_SIZE = 32
LR_STEP = 3
VAL_SIZE = 500

#######################################

# tfRecord tuner
AUTO = tf.data.experimental.AUTOTUNE

def parse_tfrecord(example):
    ''' It is strange you need to use tf.string to read in an image '''
    feature_description = {
        "rows": tf.io.FixedLenFeature([], tf.int64),
        "cols": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "spec": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)

    X = tf.io.decode_raw(
        example['spec'], out_type='float32', little_endian=True, fixed_length=None, name=None
    )

    rows = example['rows']
    cols = example['cols']
    depth = example['depth']
    X = tf.reshape(X, (rows,cols,depth))*255
    y = example["y"]
    y_hot = tf.one_hot(y, 10)
    return (X, y_hot)

##############################

def get_dataset(record_files):
    """ Loads tfRecord files
    Parameters
    ----------
    record_files : str or list of str
        paths of tfRecord files
    Returns
    -------
    tf.data.Dataset
        data loader for keras model
    """
    dataset = tf.data.TFRecordDataset(record_files, buffer_size=10000)
    dataset = (
        dataset.map(parse_tfrecord, num_parallel_calls=AUTO).cache().shuffle(10000)
    )
    dataset = dataset.prefetch(AUTO)
    return dataset

#######################################

def scheduler(epoch, lr):
    """ Scales learning rate based on current epoch """
    if epoch % LR_STEP == 0:
        return lr * 0.9
    else:
        return lr

def getModel(input_shape=(128, 250, 3)):
    # Load base model
    base_model = tf.keras.applications.EfficientNetB4(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=input_shape,
        include_top=False,
    )
    feat_ex = tf.keras.Model(base_model.input, base_model.output)

    # Add new layers
    inputs = tf.keras.Input(input_shape)
    x = feat_ex(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    yh = layers.Dense(
        10, kernel_regularizer=regularizers.l2(0.0001), activation="softmax"
    )(x)
    model = tf.keras.Model(inputs, yh)
    print(model.summary())
    return model

#######################################

if __name__ == "__main__":
    # Intialize arguments
    args = parser.parse_args()
    EPOCHS = args.epochs

    #######################################

    # Parse data folds
    folds = []
    for f in range(1, 11):
        fold = [i for i in range(1, 11) if i != f]
        folds.append((f, fold))
    print(folds)

    #######################################

    # Training loop
    total_results = []
    mel_results, mfcc_results, chroma_results = [], [], []
    for k,data_path in enumerate([data_path_mel,data_path_mfcc,data_path_chroma]):
        print("Train set: {}".format(data_path))

        for i, fold in enumerate(folds):
            print("Fold {}".format(i + 1))
            fold_index = i

            # Load datasets
            train_path = [data_path + str(i) + ".tfrec" for i in fold[1][:]]
            test_path = data_path + str(fold[0]) + ".tfrec"

            full_dataset = get_dataset(train_path)
            val_dataset = full_dataset.take(VAL_SIZE).batch(BATCH_SIZE)
            train_dataset = full_dataset.skip(VAL_SIZE).batch(BATCH_SIZE)
            test_dataset = get_dataset(test_path).batch(BATCH_SIZE)


            model = getModel()
            opt = tf.keras.optimizers.Adam(1e-4)
            model.compile(
                        optimizer=opt,
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=["accuracy"],
                    )

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath="audio_model.h5",
                        save_weights_only=True,
                        monitor="val_accuracy",
                        mode="max",
                        save_best_only=True,
                    )

            lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
            history = model.fit(train_dataset,epochs=EPOCHS,validation_data=val_dataset,
                                    callbacks=[model_checkpoint_callback, lr_callback],
                                    verbose=1,)

            # Load best weights and evaluate on test data
            model.load_weights("audio_model.h5")
            print("Evaluate on test data")
            results = model.evaluate(test_dataset)
            total_results.append(results)
            print("test loss, test acc:", results)

            # Clear graph
            tf.keras.backend.clear_session()

            if os.path.exists("audio_model.h5"):
                os.remove("audio_model.h5")

        if k == 0:
            mel_results = copy.deepcopy(total_results)
        elif k == 1:
            mfcc_results = copy.deepcopy(total_results)
        else:
            chroma_results = copy.deepcopy(total_results)

        # Save final weights and show final results
        model.save_weights(data_path.split("/")[-2]+"_weights.h5")
        print("Total Results: \n {}".format(total_results))
        total_results = []

    print("Mel: {}".format(mel_results))
    print("MFCC: {}".format(mfcc_results))
    print("Chroma: {}".format(chroma_results))

    results = {"Mel":mel_results,"MFCC":mfcc_results,"Chroma":chroma_results}
    df_results = pd.DataFrame(results)
    df_results.to_csv("results.csv")
