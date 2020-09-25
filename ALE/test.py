import unittest
from Data.data import *
from AL import algos
from engine import Engine
from Zoo import zoo

import tensorflow as tf


##############################################

class TestData(unittest.TestCase):
    @classmethod
    def setUp(cls) -> None:
        pass

    @classmethod
    def tearDown(cls) -> None:
        pass

    def test_mnistLoader1(self):
        print("\n"+"Running test_mnistLoader1 test.")

        """ Test load mnist csv raw into train, val and cache with dataManager class """
        if os.path.isfile("Data/DataSets/MNIST/train_cache.csv"):
            os.remove("Data/DataSets/MNIST/train_cache.csv")
        if os.path.isfile("Data/DataSets/MNIST/val_cache.csv"):
            os.remove("Data/DataSets/MNIST/val_cache.csv")
        if os.path.isfile("Data/DataSets/MNIST/ul_cache.csv"):
            os.remove("Data/DataSets/MNIST/ul_cache.csv")
        if os.path.isdir("Data/DataSets/MNIST/ul_cache"):
            shutil.rmtree("Data/DataSets/MNIST/ul_cache")

        dataName = "MNIST"
        split = (.2, .2, .6)  # (train, val, unlabeled)
        bins = 3
        keep_bins = True

        dataClass = ToyALoader(bins, keep_bins)  # Declare data manager class
        dataClass.parseData(split, bins, keep_bins)
        dataClass.loadCaches()
        dataClass.deleteCache()

    def test_mnistLoader2(self):
        print("\n"+"Running test_mnistLoader2 test.")

        """ Test load mnist csv raw into train, val and cache with dataManager class """
        if os.path.isfile("Data/DataSets/MNIST/train_cache.csv"):
            os.remove("Data/DataSets/MNIST/train_cache.csv")
        if os.path.isfile("Data/DataSets/MNIST/val_cache.csv"):
            os.remove("Data/DataSets/MNIST/val_cache.csv")
        if os.path.isfile("Data/DataSets/MNIST/ul_cache.csv"):
            os.remove("Data/DataSets/MNIST/ul_cache.csv")
        elif os.path.isdir("Data/DataSets/MNIST/ul_cache"):
            shutil.rmtree("Data/DataSets/MNIST/ul_cache")

        dataName = "MNIST"
        split = (.2, .2, .6)  # (train, val, unlabeled)
        bins = 1
        keep_bins = True

        dataClass = ToyALoader(bins, keep_bins)  # Declare data manager class
        dataClass.parseData(split, bins, keep_bins)
        dataClass.loadCaches()
        dataClass.deleteCache()

    def test_mnistLoader3(self):
        print("\n"+"Running test_mnistLoader3 test.")

        """ Test load mnist csv raw into train, val and cache with dataManager class """
        if os.path.isfile("Data/DataSets/MNIST/train_cache.csv"):
            os.remove("Data/DataSets/MNIST/train_cache.csv")
        if os.path.isfile("Data/DataSets/MNIST/val_cache.csv"):
            os.remove("Data/DataSets/MNIST/val_cache.csv")
        if os.path.isfile("Data/DataSets/MNIST/ul_cache.csv"):
            os.remove("Data/DataSets/MNIST/ul_cache.csv")
        elif os.path.isdir("Data/DataSets/MNIST/ul_cache"):
            shutil.rmtree("Data/DataSets/MNIST/ul_cache")

        dataName = "MNIST"
        split = (.2, .2, .6)  # (train, val, unlabeled)
        bins = 3
        keep_bins = False

        dataClass = ToyALoader(bins, keep_bins)  # Declare data manager class
        dataClass.parseData(split, bins, keep_bins)
        dataClass.loadCaches()
        dataClass.deleteCache()

    def test_mnistLoader4(self):
        print("\n"+"Running test_mnistLoader4 test.")

        """ Test load mnist csv raw into train, val and cache with dataManager class """
        if os.path.isfile("Data/DataSets/MNIST/train_cache.csv"):
            os.remove("Data/DataSets/MNIST/train_cache.csv")
        if os.path.isfile("Data/DataSets/MNIST/val_cache.csv"):
            os.remove("Data/DataSets/MNIST/val_cache.csv")
        if os.path.isfile("Data/DataSets/MNIST/ul_cache.csv"):
            os.remove("Data/DataSets/MNIST/ul_cache.csv")
        elif os.path.isdir("Data/DataSets/MNIST/ul_cache"):
            shutil.rmtree("Data/DataSets/MNIST/ul_cache")

        dataName = "MNIST"
        split = (.2, .2, .6)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = ToyALoader(bins, keep_bins)  # Declare data manager class
        dataClass.parseData(split, bins, keep_bins)
        dataClass.loadCaches()
        dataClass.deleteCache()

    def test_run_ToyA(self):
        print("\n" + "Running test_run_ToyA test.")
        # | ----------------------------
        # | 1. Select data
        # | ---------------------------

        # DataManager parameters
        split = (.1, .1, .8)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = ToyALoader(bins, keep_bins)  # Declare data manager class
        dataClass.parseData(split, bins, keep_bins)
        dataClass.loadCaches()

        # | ----------------------------
        # | 2. Select Active Learning algorithm
        # | ----------------------------

        algo = algos.leastConfidence()  # Randomly selects samples from each round's cache
        algo.reset()

        # | ----------------------------
        # | 3. Select model
        # | ----------------------------

        modelName = "ToyA_NN"  # Pick pre-made model
        metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        # | ----------------------------
        # | 4. Run algorithm and log results
        # | ----------------------------

        # Declare engine
        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)

        # Initial training of model on original training data
        engine.initialTrain(epochs=1, batch_size=32, val=True, plot=False)

        # Run active learning algo
        # Round is how many times the active learning algo samples
        # cycles is how many epochs the model is retrained each time a round occurs of sampling
        engine.run(rounds=1, cycles=1, batch_size=32, val=True, plot=False)
        #engine.saveLog(path="test_log.csv")
        dataClass.deleteCache()

    def test_run_MNIST(self):
        print("\n" + "Running test_run_ToyA test.")
        # | ----------------------------
        # | 1. Select data
        # | ---------------------------

        # DataManager parameters
        split = (.1, .1, .8)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)  # Declare data manager class
        dataClass.parseData(split, bins, keep_bins)
        dataClass.loadCaches()

        # | ----------------------------
        # | 2. Select Active Learning algorithm
        # | ----------------------------

        algo = algos.leastConfidence()  # Randomly selects samples from each round's cache
        algo.reset()

        # | ----------------------------
        # | 3. Select model
        # | ----------------------------

        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.KLDivergence()]
        zk = zoo.zooKeeper(modelName, show_model=True, metrics=metrics)  # Load model and compile

        # | ----------------------------
        # | 4. Run algorithm and log results
        # | ----------------------------

        # Declare engine
        sample_size = 50
        engine = Engine(algo, dataClass, zk, sample_size)

        # Initial training of model on original training data
        engine.initialTrain(epochs=1, batch_size=32, val=True, plot=False)

        # Run active learning algo
        # Round is how many times the active learning algo samples
        # cycles is how many epochs the model is retrained each time a round occurs of sampling
        engine.run(rounds=1, cycles=1, batch_size=32, val=True, plot=False)
        #engine.saveLog(path="test_log.csv")
        dataClass.deleteCache()

##############################################

if __name__ == "__main__":
    unittest.main()