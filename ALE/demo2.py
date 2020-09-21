"""

Purpose: This script demonstrates using the pre-built mnist data set and loader, random
sampling, and the pre-build mnistCNN network.

To use: remove files and folders for mnist cache.
"""


from Data.data import *
from AL import algos
from engine import Engine
from Zoo import zoo


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
    dataName = "MNIST"
    split = (.2, .2, .6)  # (train, val, unlabeled)
    bins = 1
    keep_bins = False

    dataClass = DataManager(dataName)  # Declare data manager class
    dataClass.parseData(split, bins, keep_bins)  # Parse MNIST data to custom split and rounds
    dataClass = mnistLoader(bins, keep_bins)  # Reload custom dataClass for MNIST with new custom split and rounds

    # | ----------------------------
    # | 2. Select Active Learning algorithm
    # | ----------------------------

    algo = algos.uniformSample()  # Randomly selects samples from each round's cache
    algo.reset()

    # | ----------------------------
    # | 3. Select model
    # | ----------------------------

    modelName = "mnistCNN"  # Pick pre-made model
    zk = zoo.zooKeeper(dataName, modelName, show_model=True)  # Load model and compile

    # | ----------------------------
    # | 4. Run algorithm and log results
    # | ----------------------------

    # Declare engine
    sample_size = 100
    engine = Engine(algo, dataClass, zk, sample_size)

    # Initial training of model on original training data
    engine.initialTrain(epochs=2, batch_size=64, val=True, plot=True)

    # Run active learning algo
    # Round is how many times the active learning algo samples
    # cycles is how many epochs the model is retrained each time a round occurs of sampling
    engine.run(rounds=2, cycles=2, batch_size=64, val=True, plot=True)

    # | ----------------------------
    # | Done
    # | ----------------------------

    print("Finshed running")


if __name__ == "__main__":
    main()
