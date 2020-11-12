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
    split = (.2, .3, .5)  # (train, val, unlabeled)
    bins = 1
    keep_bins = False

    dataClass = cifar10Loader(bins, keep_bins)  # Declare data manager class
    #dataClass.parseData(split, bins, keep_bins)
    dataClass.loadCaches()

    # | ----------------------------
    # | 2. Select Active Learning algorithm
    # | ----------------------------

    #algo = algos.DALratio(input_dim=120)
    #algo = algos.AADA(input_dim=120)
    #algo = algos.DAL(input_dim=256)
    #algo = algos.DALOC(input_dim=120)
    #algo = algos.clusterDAL(input_dim=120)
    #algo = algos.VAE(input_dim=120,codings_size=20,Beta=1)
    #algo = algos.MemAE_AL(input_dim=256, codings_size=15)
    #algo = algos.MemAE_Binary_AL(input_dim=120, codings_size=15)
    #algo = algos.marginConfidence()
    algo = algos.uniformSample()
    algo.reset()

    # | ----------------------------
    # | 3. Select model
    # | ----------------------------

    modelName = "cifar10CNN"  # Pick pre-made model
    metrics = [tf.keras.metrics.Accuracy(), tf.keras.metrics.KLDivergence()]
    #metrics = [tf.keras.metrics.MeanSquaredError()]
    zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

    # | ----------------------------
    # | 4. Run algorithm and log results
    # | ----------------------------

    # Declare engine
    sample_size = 5000
    engine = Engine(algo, dataClass, zk, sample_size)

    # Initial training of model on original training data
    #engine.initialTrain(epochs=20, batch_size=32, val=True, plot=True)

    #engine.saveModel("cifar10_model")
    engine.loadModel("cifar10_model")

    # Run active learning algo
    # Round is how many times the active learning algo samples
    # cycles is how many epochs the model is retrained each time a round occurs of sampling
    engine.run(rounds=2, cycles=30, batch_size=32, val=True, plot=True)
    engine.saveLog(path="test_log.csv")
    #dataClass.deleteCache()

    # | ----------------------------
    # | Done
    # | ----------------------------


if __name__ == "__main__":
    main()
