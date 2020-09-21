import unittest
import os
import shutil
from Data import data


##############################################

class TestData(unittest.TestCase):
    @classmethod
    def setUp(cls) -> None:
        pass

    @classmethod
    def tearDown(cls) -> None:
        pass

    def test_load_data_py(self):
        print("\n"+"Running test_load_data_py test.")

        x = data.load_data_py()
        self.assertEqual(x,"Loaded data.py")

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
        rounds = 3
        keep_bins = True

        dataClass = data.DataManager(dataName)  # Declare data manager class
        dataClass.parseData(split, rounds, keep_bins)  # Parse MNIST data to custom split and rounds
        dataClass = data.mnistLoader(rounds, True)  # Reload custom dataClass for MNIST with new custom split and rounds

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
        rounds = 1
        keep_bins = True

        dataClass = data.DataManager(dataName)  # Declare data manager class
        dataClass.parseData(split, rounds, keep_bins)  # Parse MNIST data to custom split and rounds
        dataClass = data.mnistLoader(rounds, True)  # Reload custom dataClass for MNIST with new custom split and rounds


##############################################

if __name__ == "__main__":
    unittest.main()