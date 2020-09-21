import os
import pandas as pd
import numpy as np
from math import floor


#######################################################

def load_data_py():
    """ Test function to see if data.py is loading properly """
    return "Loaded data.py"


#######################################################

class DataManager:
    # Class variables
    # z = 1

    def __init__(self, dataset_name, bins=1):

        assert isinstance(dataset_name, str), "dataset_name must be a string"
        self.dataset_name = dataset_name
        self.n_total_samples = 0
        self.bins = bins  # Number of unlabeled caches corresponding to each round of retraining
        self.keep_bins = False

        # Check if data set is in Data/DataSet
        self.checkDataSet()
        self.data_path = "Data/DataSets/" + self.dataset_name

        # List to keep track of what data is used where
        self.u_cache = []  # Keeps running track of all data
        self.train_cache = []  # Keeps track of what the model is allowed to see during training
        self.val_cache = []  # Keep track of what data to use for validation
        self.unlabeled_cache = []  # Keep tracks of what data can't be used for training since unlabled

    def clearCache(self):
        """ Resets values in caches """
        self.train_cache, self.val_cache = [], []
        self.unlabeled_cache, self.u_cache = [], []

    def checkDataSet(self):
        """ Checks if passed dataset name to class has matching data folder in Data/DataSets """
        # Get list of data folders in Data/DataSets
        folder_DataSets = [f.path for f in os.scandir("./Data/DataSets") if f.is_dir()]
        folder_DataSets = [f.split("/")[-1] for f in folder_DataSets]

        try:
            i = folder_DataSets.index(self.dataset_name)
        except ValueError:
            print("ERROR: The passed data set name, \"{}\" , to the class DataLoader is not in Data/DataSets.".format(
                self.dataset_name))
            print("The following data sets are in Data/DataSets: {}".format(folder_DataSets))

    def parseData(self, split, bins=1, keep_bins=False):
        """
        Created three files (or two files and a folder with files if keep_bins == True) that
        are .csv that states what data (based on unique ID) belong to each cache

        split : set
            states what percent of data to place in train, val, and ul cache
        bins : int
            states how many ul cache bins to make
        keep_bins : bool
            states if to role over previous ul cache to current
        """

        print("-" * 20)
        print("Parsing raw data")
        print("-" * 20)

        # Check if bins is an int
        if isinstance(bins, int) != True:
            raise ValueError('bins is not of type int, bins: {}'.format(bins))
        self.bins = bins

        # Check if split was passed properly
        if sum(split) != 1:
            print("ERROR: Sum of split does not sum to 1.")
            exit()
        if len(split) < 1 or len(split) > 4:
            print("ERROR: Improper length of split")
            exit()

        # Set keep_bins as bool passed
        self.keep_bins = keep_bins

        # Open data tab
        self.data_tab = pd.read_csv(self.data_path + "/raw_data/data_tab.csv")

        # Place copy of ID col as list in u_cache
        self.u_cache = self.data_tab["ID"].tolist()

        # Place ID copies in respective cache
        self.n_total_samples = len(self.u_cache)
        # Fill train cache
        n_train = floor(self.n_total_samples * split[0])
        self.train_cache = self.u_cache[:n_train]
        # Fill val cache
        n_val = floor(self.n_total_samples * (split[0] + split[1]))
        self.val_cache = self.u_cache[n_train:n_val]
        # Fill unlabeled cache
        self.unlabeled_cache = self.u_cache[n_val:]

        # Make files to keep track of what is in each cache

        # Save unlabeled cache
        if self.bins > 1:
            cache_size_track = []
            p_bin_cache_df = None
            os.mkdir(self.data_path + "/ul_cache")
            n_samples_bin = floor(len(self.unlabeled_cache) / bins)
            if bins * n_samples_bin != len(self.unlabeled_cache):
                r = len(self.unlabeled_cache) - bins * n_samples_bin
            else:
                r = 0

            for i in range(bins):
                bin_cache = self.unlabeled_cache[i * n_samples_bin:(i + 1) * n_samples_bin]
                if i == (bins - 1) and r != 0:
                    bin_cache = self.unlabeled_cache[bins * n_samples_bin:]
                bin_cache_df = self.data_tab.loc[self.data_tab['ID'].isin(bin_cache)]
                bin_cache_df = bin_cache_df.reset_index(drop=True)

                # Add previous cache if keep_bins == True
                if keep_bins == True and i >= 1:
                    bin_cache_df = p_bin_cache_df.append(bin_cache_df)
                    bin_cache_df = bin_cache_df.reset_index(drop=True)

                bin_cache_df.to_csv(self.data_path + "/ul_cache" + "/ul_cache_" + str(i) + ".csv")
                cache_size_track.append(len(bin_cache_df))
                if keep_bins == True:
                    p_bin_cache_df = bin_cache_df
        else:
            bin_cache_df = self.data_tab.loc[self.data_tab['ID'].isin(self.unlabeled_cache)]
            bin_cache_df = bin_cache_df.reset_index(drop=True)
            bin_cache_df.to_csv(self.data_path + "/ul_cache" + ".csv")

        # Save train cache
        train_cache_df = self.data_tab.loc[self.data_tab['ID'].isin(self.train_cache)]
        train_cache_df = train_cache_df.reset_index(drop=True)
        train_cache_df.to_csv(self.data_path + "/train_cache_" + ".csv")

        # Save val cache
        val_cache_df = self.data_tab.loc[self.data_tab['ID'].isin(self.val_cache)]
        val_cache_df = val_cache_df.reset_index(drop=True)
        val_cache_df.to_csv(self.data_path + "/val_cache_" + ".csv")

        print("\n")
        print("Data split:")
        print("{} samples in training cache.".format(len(train_cache_df)))
        print("{} samples in validation cache.".format(len(val_cache_df)))

        if self.bins == 1:
            print("{} samples in unlabled cache.".format(len(bin_cache_df)))
        else:
            for i, s in enumerate(cache_size_track):
                print("Unlabeled cache {} has {} samples.".format(i, s))
        print("\n")

        # Clear cache list values
        self.clearCache()

    def loadCaches(self):
        """
        Reload cache (train, val, and ul_cache)
        """
        # Load unlabeled cache ids
        if self.bins == 1:
            temp_cache_df = pd.read_csv(self.data_path + "/ul_cache" + ".csv")
            self.unlabeled_cache = temp_cache_df["ID"].tolist()
        else:
            for i in range(self.bins):
                temp_cache_df = pd.read_csv(self.data_path + "/ul_cache/ul_cache_" + str(i) + ".csv")
                self.unlabeled_cache.append(temp_cache_df["ID"].tolist())

        # Load train cache ids
        temp_cache_df = pd.read_csv(self.data_path + "/train_cache_" + ".csv")
        self.train_cache = temp_cache_df["ID"].tolist()

        # Load val cache ids
        temp_cache_df = pd.read_csv(self.data_path + "/val_cache_" + ".csv")
        self.val_cache = temp_cache_df["ID"].tolist()

    def sampleUCache(self, method, bin_n=None):
        pass


#######################################################

class CustomGenerator(DataManager):
    def __init__(self, dataset_name, bins=1):
        super().__init__(dataset_name, bins)
        dummy = 0

    def getBatch(self, ids):
        # TODO: Make function that selects from right cache and loads data for Keras model
        # TODO: Make custom data loader function that loads data based on type (csv, image, etc) and reshapes
        pass


#######################################################

class mnistLoader(DataManager):
    """
    mnistLoader Documentation:
    --------------------------

    Purpose
    ----------
    Load data as needed from the mnist dataset. Currently, I just load the whole
    data set into memory. Not efficient at all yet, currently unaware of alternative
    method. Need to keep looking into this issue. mnistLoader is a child instance of the class
    DataManager. DataManager is disgustingly large and is a mental pain ... I'll go back and clean it up.

    Attributes
    ----------
    data_df : pd.DataFrame()
        Stores dataframe of mnist data. This is unique to this class. Each child class of DataManager
        is going to have a unique feature such as this due to using different types of data.

    Methods
    -------
    getBatch(self, cache_ids: list):
        grabs row of data from data_df based on id's in passed list. Converts into
        numpy arrays. Makes input matrix and one hot encoded array as target value.
        Returns processed input and target data.

    """

    def __init__(self, bins=1, keep_bins=False):
        super().__init__("MNIST", bins)

        self.keep_bins = keep_bins
        self.loadCaches()
        self.data_df = pd.read_csv(self.data_path + "/raw_data/mnist.csv", iterator=True, chunksize=5000)
        self.data_df = pd.concat(self.data_df, ignore_index=True)

    def getBatch(self, cache_ids: list):
        # Select data based off of passed cache_ids
        batchData = self.data_df.loc[self.data_df['ID'].isin(cache_ids)]
        y = batchData[["Label"]].values
        X = batchData.iloc[:, 2:].values

        # Reshape input X
        X = X.reshape(X.shape[0], 28, 28, 1)
        X = X / 255  # Scale input data between [0,1]

        # One hot encode targets y
        y_onehot = np.zeros((y.shape[0], 10))
        for i in range(y_onehot.shape[0]):
            y_onehot[i, y[i]] = 1

        return X, y_onehot
