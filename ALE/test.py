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

        split = (.2, .2, .6)  # (train, val, unlabeled)
        bins = 1
        keep_bins = True

        dataClass = mnistLoader(bins, keep_bins)  # Declare data manager class
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

        split = (.2, .2, .6)  # (train, val, unlabeled)
        bins = 3
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)  # Declare data manager class
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

        split = (.2, .2, .6)  # (train, val, unlabeled)
        bins = 1
        keep_bins = False

        dataClass = mnistLoader(bins, keep_bins)  # Declare data manager class
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
        dataClass.deleteCache()

    def test_run_MNIST(self):
        print("\n" + "Running test_run_MNIST test.")
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
        dataClass.deleteCache()

    def test_customAlgo(self):
        """ Check to see if custom children class AL algo are created with accordance to inheritance """
        # Test passive learning
        algo = algos.uniformSample()
        self.assertTrue(algos.alAlgo.__subclasshook__(algo))

        # Test ratio confidence
        algo = algos.ratioConfidence()
        self.assertTrue(algos.alAlgo.__subclasshook__(algo))

        # Test least confidence
        algo = algos.leastConfidence()
        self.assertTrue(algos.alAlgo.__subclasshook__(algo))

        # Test margin confidence
        algo = algos.marginConfidence()
        self.assertTrue(algos.alAlgo.__subclasshook__(algo))

        # Test entropy
        algo = algos.entropy()
        self.assertTrue(algos.alAlgo.__subclasshook__(algo))

    def test_customModel(self):
        """ Check to see if custom children class zoo.customModel are created with accordance to inheritance """
        # Test mnsitCNN
        modelName = "mnistCNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.Accuracy()]
        zk = zoo.zooKeeper(modelName, show_model=False, metrics=metrics)
        self.assertTrue(zoo.customModel.__subclasshook__(zk))

        # Test ToyA_NN
        modelName = "ToyA_NN"  # Pick pre-made model
        metrics = [tf.keras.metrics.Accuracy()]
        zk = zoo.zooKeeper(modelName, show_model=False, metrics=metrics)
        self.assertTrue(zoo.customModel.__subclasshook__(zk))

        # Test ToyA_NN
        modelName = "cifar10CNN"  # Pick pre-made model
        metrics = [tf.keras.metrics.Accuracy()]
        zk = zoo.zooKeeper(modelName, show_model=False, metrics=metrics)
        self.assertTrue(zoo.customModel.__subclasshook__(zk))

    def test_zooKeeperRaiseNoModel(self):
        """ This test checks if error is raised from zooKeeper.getmodel() for non existent model"""
        self.assertRaises(ImportError,zoo.zooKeeper,"dummy_model")


##############################################

if __name__ == "__main__":
    unittest.main()