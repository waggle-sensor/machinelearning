"""
python train_model.py --n_gpu 8
"""


#######################################

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

#######################################

data_path = "UrbanSound8k/tfRecords/fold"
data_aug_path = "UrbanSound8k/tfRecords_aug/fold"

BATCH_SIZE = 32
LR_STEP = 1
EPOCHS = 4
VAL_SIZE = 500

#######################################

AUTO = tf.data.experimental.AUTOTUNE

def parse_tfrecord(example):
    """ It is strange you need to use tf.string to read in an image """
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
    dataset = tf.data.TFRecordDataset(record_files, buffer_size=10000)
    dataset = (
        dataset.map(parse_tfrecord, num_parallel_calls=AUTO).cache().shuffle(10000)
    )
    dataset = dataset.prefetch(AUTO)
    return dataset


#######################################


def scheduler(epoch, lr):
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
    args = parser.parse_args()

    if args.n_gpu > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        N_GPU = args.n_gpu
        BATCH_SIZE = BATCH_SIZE * N_GPU

    #######################################

    folds = []
    for f in range(1, 11):
        fold = [i for i in range(1, 11) if i != f]
        folds.append((f, fold))
    print(folds)

    #######################################

    total_results = []
    for i, fold in enumerate(folds):
        print("Fold {}".format(i + 1))
        fold_index = i
        

        train_path = [data_aug_path + str(i) + ".tfrec" for i in fold[1][:]]
        test_path = data_path + str(fold[0]) + ".tfrec"

        full_dataset = get_dataset(train_path)
        val_dataset = full_dataset.take(VAL_SIZE).batch(BATCH_SIZE)
        train_dataset = full_dataset.skip(VAL_SIZE).batch(BATCH_SIZE)
        test_dataset = get_dataset(test_path).batch(BATCH_SIZE)

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

        model.load_weights("audio_model.h5")

        print("Evaluate on test data")
        results = model.evaluate(test_dataset)
        total_results.append(results)
        print("test loss, test acc:", results)

        if os.path.exists("audio_model.h5"):
            os.remove("audio_model.h5")

    model.save_weights("final_weights.h5")
    print("Total Results: \n {}".format(total_results))
