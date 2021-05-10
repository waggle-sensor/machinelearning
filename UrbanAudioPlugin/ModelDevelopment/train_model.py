"""
train_model.py

Description: code used to train neural network over UrbanSound8k processed data.

Takes three arguments: --n_gpu (int > 0) --epochs (int > 0) --use_aug (True,False)
Ex: python train_model.py --n_gpu 8 --epochs 5 --use_aug True
"""

#######################################

# Import modules
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

#######################################

parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--use_aug", type=bool)

#######################################

# Declare path to data
data_path = "UrbanSound8k/tfRecords/fold"
data_aug_path = "UrbanSound8k/tfRecords_aug/fold"

# Declare global variables
BATCH_SIZE = 32
LR_STEP = 3
VAL_SIZE = 500

#######################################

AUTO = tf.data.experimental.AUTOTUNE

def parse_tfrecord(example):
    """ Decodes tfRecord file into data samples  """
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_png(example["image"], channels=3)
    image = example["image"]
    target = example["target"]
    target_hot = tf.one_hot(target, 10)
    return (image, target_hot)


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


def getEfficientNetB4(input_shape=(128, 250, 3)):
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

    # Set up multiple gpus if asked
    if args.n_gpu > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        N_GPU = args.n_gpu
        BATCH_SIZE = BATCH_SIZE * N_GPU

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
    for i, fold in enumerate(folds):
        print("Fold {}".format(i + 1))
        fold_index = i

        # Load datasets
        if args.use_aug == True:
            train_path = [data_aug_path + str(i) + ".tfrec" for i in fold[1][:]]
        else:
            train_path = [data_path + str(i) + ".tfrec" for i in fold[1][:]]
        test_path = data_path + str(fold[0]) + ".tfrec"

        full_dataset = get_dataset(train_path)
        val_dataset = full_dataset.take(VAL_SIZE).batch(BATCH_SIZE)
        train_dataset = full_dataset.skip(VAL_SIZE).batch(BATCH_SIZE)
        test_dataset = get_dataset(test_path).batch(BATCH_SIZE)

        # Declare model and train
        if args.n_gpu > 1:
            print("Using mirrored strategy!")
            with mirrored_strategy.scope():
                model = getEfficientNetB4()
                opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
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

                history = model.fit(
                    train_dataset,
                    epochs=EPOCHS,
                    validation_data=val_dataset,
                    callbacks=[model_checkpoint_callback, lr_callback],
                    verbose=1,
                )

        else:
            model = getEfficientNetB4()
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

            history = model.fit(
                train_dataset,
                epochs=EPOCHS,
                validation_data=val_dataset,
                callbacks=[model_checkpoint_callback, lr_callback],
                verbose=1,
            )

        # Load best weights and evaluate on test data
        model.load_weights("audio_model.h5")
        print("Evaluate on test data")
        results = model.evaluate(test_dataset)
        total_results.append(results)
        print("test loss, test acc:", results)

        if os.path.exists("audio_model.h5"):
            os.remove("audio_model.h5")

    # Save final weights and show final results
    model.save_weights("final_weights.h5")
    print("Total Results: \n {}".format(total_results))
