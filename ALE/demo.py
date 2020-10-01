"""

Purpose: This script demonstrates using the pre-built mnist data set and loader, random
sampling, and the pre-build mnistCNN network.

To use: remove files and folders for mnist cache.
"""

from Data.data import *
from AL import algos
from engine import Engine
from Zoo import zoo

import tensorflow as tf


#######################################################


def main():
    """
    How to use:

    # 1. Select data

    # 2. Select Active Learning algorithm

    # 3. Select model

    # 4. Run algorithm and log results

    """

    # | ----------------------------
    # | 1. Select data
    # | ---------------------------

    # DataManager parameters
    split = (.1, .1, .8)  # (train, val, unlabeled)
    bins = 1
    keep_bins = False

    dataClass = cifar10Loader(bins, keep_bins)  # Declare data manager class
    dataClass.parseData(split, bins, keep_bins)
    dataClass.loadCaches()

    # | ----------------------------
    # | 2. Select Active Learning algorithm
    # | ----------------------------

    algo = algos.entropy()  # Randomly selects samples from each round's cache
    algo.reset()

    # | ----------------------------
    # | 3. Select model
    # | ----------------------------

    modelName = "MobileNet"  # Pick pre-made model
    metrics = [tf.keras.metrics.Accuracy(), tf.keras.metrics.KLDivergence()]
    zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics,dataset="CIFAR10")  # Load model and compile

    # | ----------------------------
    # | 4. Run algorithm and log results
    # | ----------------------------

    # Declare engine
    sample_size = 100
    engine = Engine(algo, dataClass, zk, sample_size)
    input("press enter to continue")

    # Initial training of model on original training data
    engine.initialTrain(epochs=15, batch_size=32, val=True, plot=True)

    # Run active learning algo
    # Round is how many times the active learning algo samples
    # cycles is how many epochs the model is retrained each time a round occurs of sampling
    engine.run(rounds=1, cycles=1, batch_size=32, val=True, plot=True)
    engine.saveLog(path="test_log.csv")
    dataClass.deleteCache()

    # | ----------------------------
    # | Done
    # | ----------------------------


if __name__ == "__main__":
    main()
