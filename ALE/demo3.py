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
    dataName = "ToyA"
    split = (.05, .05, .9)  # (train, val, unlabeled)
    bins = 5
    keep_bins = True

    dataClass = DataManager(dataName)  # Declare data manager class
    dataClass.parseData(split, bins, keep_bins)  # Parse MNIST data to custom split and rounds
    dataClass = ToyALoader(bins, keep_bins)  # Reload custom dataClass for MNIST with new custom split and rounds

    # X, y = dataClass.getBatch(list(range(0,10)))

    # | ----------------------------
    # | 2. Select Active Learning algorithm
    # | ----------------------------

    algo = algos.uniformSample()  # Randomly selects samples from each round's cache
    algo.reset()

    # | ----------------------------
    # | 3. Select model
    # | ----------------------------

    modelName = "ToyA_NN"  # Pick pre-made model
    metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.KLDivergence()]
    zk = zoo.zooKeeper(dataName, modelName, show_model=True, metrics=metrics)  # Load model and compile

    # | ----------------------------
    # | 4. Run algorithm and log results
    # | ----------------------------

    # Declare engine
    sample_size = 50
    engine = Engine(algo, dataClass, zk, sample_size)

    # Initial training of model on original training data
    engine.initialTrain(epochs=10, batch_size=16, val=True, plot=True)

    # Run active learning algo
    # Round is how many times the active learning algo samples
    # cycles is how many epochs the model is retrained each time a round occurs of sampling
    engine.run(rounds=5, cycles=2, batch_size=16, val=True, plot=True)

    # | ----------------------------
    # | Done
    # | ----------------------------

    print("Finshed running")


if __name__ == "__main__":
    main()
