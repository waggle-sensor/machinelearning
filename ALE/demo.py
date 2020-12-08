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
    split = (.0735, .2, .7265)  # (train, val, unlabeled)
    bins = 1
    keep_bins = False

    dataClass = monkeyLoader(bins, keep_bins)  # Declare data manager class
    #dataClass.parseData(split, bins, keep_bins)
    dataClass.loadCaches()

    # | ----------------------------
    # | 2. Select Active Learning algorithm
    # | ----------------------------

    #algo = algos.DALratio(input_dim=120)
    #algo = algos.AADA(input_dim=120)
    #algo = algos.DAL(input_dim=120)
    #algo = algos.DALOC(input_dim=120)
    #algo = algos.clusterDAL(input_dim=120)
    #algo = algos.VAE(input_dim=120,codings_size=20,Beta=1)
    #algo = algos.MemAE_AL(input_dim=120, codings_size=15)
    #algo = algos.MemAE_Binary_AL(input_dim=120, codings_size=15)
    #algo = algos.marginConfidence()
    #algo = algos.uniformSample()

    # Add adjust p value each round
    algo = algos.clusterMargin(n_cluster=50, p=.6, sub_sample=150)
    algo.reset()

    # | ----------------------------
    # | 3. Select model
    # | ----------------------------

    modelName = "monkeyCNN"  # Pick pre-made model
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.KLDivergence()]
    zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

    # | ----------------------------
    # | 4. Run algorithm and log results
    # | ----------------------------

    # Declare engine
    sample_size = 50
    engine = Engine(algo, dataClass, zk, sample_size)

    # Initial training of model on original training data
    #engine.initialTrain(epochs=30, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=True)

    #engine.saveModel("monkey_model")
    engine.loadModel("monkey_model")

    # Run active learning algo
    # Round is how many times the active learning algo samples
    # cycles is how many epochs the model is retrained each time a round occurs of sampling
    engine.run(rounds=10, cycles=20, batch_size=32, val=True, val_track="sparse_categorical_accuracy", plot=True)
    engine.saveLog(path="cluster_monkey_log.csv")
    #dataClass.deleteCache()

    # | ----------------------------
    # | Done
    # | ----------------------------


if __name__ == "__main__":
    main()
