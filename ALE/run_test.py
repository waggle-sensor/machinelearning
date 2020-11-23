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

    for i in range(0,15):
        np.random.seed(i+1)
        # DataManager parameters
        split = (.00143, .2, .79857)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)  # Declare data manager class
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
        #algo = algos.uniformSample()
        #algo = algos.marginConfidence()
        #algo = algos.clusterMargin(n_cluster=100, p=.8, sub_sample=500)
        #algo = algos.VAE(input_dim=120, codings_size=15,Beta=1)
        algo = algos.clusterMargin(n_cluster=100, p=.8, sub_sample=300)
        algo.reset()

        # | ----------------------------
        # | 3. Select model
        # | ----------------------------

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.Accuracy(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        # | ----------------------------
        # | 4. Run algorithm and log results
        # | ----------------------------

        # Declare engine
        sample_size = 100
        engine = Engine(algo, dataClass, zk, sample_size)

        # Initial training of model on original training data
        #engine.initialTrain(epochs=15, batch_size=32, val=True, val_track="accuracy", plot=False)

        #engine.saveModel("mnist_model")
        engine.loadModel("mnist_model")

        # Run active learning algo
        # Round is how many times the active learning algo samples
        # cycles is how many epochs the model is retrained each time a round occurs of sampling
        engine.run(rounds=8, cycles=15, batch_size=32, val=True, val_track="accuracy", plot=False)
        engine.saveLog(path="results/mnist/marginCluster/cm_"+str(i)+"_log.csv")
        #dataClass.deleteCache()


    # | ----------------------------
    # | Done
    # | ----------------------------


if __name__ == "__main__":
    main()
