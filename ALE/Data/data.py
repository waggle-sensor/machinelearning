# Import modules
import os
import abc
import shutil
import pandas as pd
import numpy as np
from math import floor

from PIL import Image
from skimage.io import imread
from skimage.transform import resize

#######################################################

"""

How to make a new data set: 
1. Obtain new data (case dependent)
2. Place raw orginal data in folder named "abc" (abc is an example name, make it whatever you want)
    within Data/DataSets/abc/raw_data
3. Create two csv files: 
        I. data_tab.csv whose columns are unique ids that match to each sample and each sample's target values
        II. a csv file named abc.csv that has each samples id and also input values in each row (in cases where
                input data can not be formatted in such a manner, place raw data in folder within raw_data named 
                abc where each individual data sample is named it's respective id



"""


#######################################################

def load_data_py():
    """ Test function to see if data.py is loading properly """
    return "Loaded data.py"


#######################################################

class DataManager(metaclass=abc.ABCMeta):
    """
    DataManger Documentation:
    --------------------------

    Purpose
    ----------
    DataManger is the parent class for custom and pre-made data loaders. DataManger serves
    critical purposes: check if requested data is present in Data/Datasets, make new data
    from original data to be used for Active Learning (initial training data, validation data,
    unlabeled samples (could be one cache of unlabeled data or multiple sub classes of unlabeled data), and
    track what samples are removed and replaced in different caches based on the Active Learning algorithms
    choices.

    Arguments
    _________
    dataset_name : str
        dataset_name is a string that is used to select what data to be used by DataManager. In __init__ section
        B., logic is shown on how DataManger checks if a data set is present and formatted in the proper way.

    bins : int
        bins is a int that states how many unique caches should be made for the unlabeled data. For example,
        say that there are 300 unlabeled examples and bins = 3. Then three unlabeled caches (u_cache_0, u_cache_1,
        u_cache_2) will be made each containing one hundred unique samples from the larger set of 300 unlabeled
        samples.

    keep_bins : bool
        keep_bins is a bool that determines whether or not to place unselected values from current round
        into the next round's pool. For example, say current round's pool is [a,b,d] and next round's pool
        is [d,e,f]. Say keep_bins==True and 'a' was chosen, then the next round's pool would be updated to
        [b,c,d,e,f].

    Attributes
    ----------
    data_path : str
        path to folder where data is stored within ./Data/

    u_cache : list
        list to store all ids of samples

    train_cache : list
        list to store ids of what samples belong to the training set

    val_cache : list
        list to store ids of what samples belong to the validation set

    unlabeled_cache : list
        list to store ids of what samples belong to the unlabeled set that can be selected from during
        active learning. NOTE! unlabeled cache can be a list of list where the sublist are the cache ids
        that are produced from bins > 1 where multiple unlabeled caches are made.

    Methods
    -------
    clearCache(self):
        This method sets u_cache, train_cache, val_cache, and unlabeled_cache back to empty list []

    checkDataSet(self):
       Checks if passed data set name has respective folder

    parseData(self, split, bins=1, keep_bins=False):
        Makes train, validation, and unlabeled sampling cache csv files that contain
        respective sample ids

    loadCaches(self):
        Reads each cache's csv file and stores respective ids in train_cache, val_cache, unlabeled_cache

    deleteCache(self):
        Deletes each cache's csv files. This may be needed if you desire to re-run an test from scratch

    """

    def __init__(self, dataset_name, bins=1, keep_bins=None):

        # A. Fields of DataManger class
        self.n_total_samples = 0
        self.bins = bins  # Number of unlabeled caches corresponding to each round of retraining
        self.keep_bins = keep_bins  # Push previous rounds unused samples to next round

        # B. Check if data set is in Data/DataSet
        assert isinstance(dataset_name, str), "dataset_name must be a string"
        self.dataset_name = dataset_name
        self.checkDataSet()
        self.data_path = "Data/DataSets/" + self.dataset_name

        # C. List to keep track of what data is used where
        self.u_cache = []  # Keeps running track of all data
        self.train_cache = []  # Keeps track of what the model is allowed to see during training
        self.val_cache = []  # Keep track of what data to use for validation
        self.unlabeled_cache = []  # Keep tracks of what data can't be used for training since unlabled

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'getBatch') and
                callable(subclass.getBatch) or
                NotImplemented)

    @abc.abstractmethod
    def getBatch(self, cache_ids: list):
        """ Load data and pre-process (reshape, scale, etc) for the model """
        raise NotImplementedError

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

        print("\n")
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
        if self.keep_bins == None:
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

    def deleteCache(self):
        """
        Delete cache (train, val, and ul_cache)
        """
        # Delete unlabeled cache csv
        if self.bins == 1:
            os.remove(self.data_path + "/ul_cache" + ".csv")
        else:
            shutil.rmtree(self.data_path + "/ul_cache")

        # Delete train cache csv
        os.remove(self.data_path + "/train_cache_" + ".csv")

        # Delete val cache csv
        os.remove(self.data_path + "/val_cache_" + ".csv")


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

class ToyALoader(DataManager):
    """
    ToyALoader Documentation:
    --------------------------

    Purpose
    ----------
    Load data as needed from the ToyA dataset. Currently, I just load the whole
    data set into memory. Not efficient at all yet, currently unaware of alternative
    method. Need to keep looking into this issue. ToyA is a child instance of the class
    DataManager. DataManager is disgustingly large and is a mental pain ... I'll go back and clean it up.

    Attributes
    ----------
    data_df : pd.DataFrame()
        Stores dataframe of ToyA data. This is unique to this class. Each child class of DataManager
        is going to have a unique feature such as this due to using different types of data.

    Methods
    -------
    getBatch(self, cache_ids: list):
        grabs row of data from data_df based on id's in passed list. Converts into
        numpy arrays. Makes input matrix and one hot encoded array as target value.
        Returns processed input and target data.

    """

    def __init__(self, bins=1, keep_bins=False):
        super().__init__("ToyA", bins, keep_bins)

        # For this data set, it is possible to read in all data at one
        self.data_df = pd.read_csv(self.data_path + "/raw_data/toyA.csv", iterator=True, chunksize=5000)
        self.data_df = pd.concat(self.data_df, ignore_index=True)

    def getBatch(self, cache_ids: list):
        """
        Purpose
        _______
        Receives from algo class list of sample ids to get from memory or RAM

        Arguments
        __________
        cache_ids : list
            list contains ids of what data samples to send back

        Returns
        _______
        X : np.array
            Input values for model
        y : Target values for model

       """

        # Select data based off of passed cache_ids
        batchData = self.data_df.loc[self.data_df['ID'].isin(cache_ids)]
        y = batchData[["y"]].values
        X = batchData.iloc[:, :2].values

        # One hot encode targets y
        y_onehot = np.zeros((y.shape[0], 2))
        for i in range(y_onehot.shape[0]):
            if y[i] == -1:
                y_onehot[i, 0] = 1
            else:
                y_onehot[i, 1] = 1

        return X, y_onehot


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
        super().__init__("MNIST", bins, keep_bins)

        # For this data set, it is possible to read in all data at one
        self.data_df = pd.read_csv(self.data_path + "/raw_data/mnist.csv", iterator=True, chunksize=5000)
        self.data_df = pd.concat(self.data_df, ignore_index=True)

    def getBatch(self, cache_ids: list):
        """
        Purpose
        _______
        Receives from algo class list of sample ids to get from memory or RAM

        Arguments
        __________
        cache_ids : list
            list contains ids of what data samples to send back

        Returns
        _______
        X : np.array
            Input values for model
        y : Target values for model

        """

        # Select data based off of passed cache_ids
        batchData = self.data_df.loc[self.data_df['ID'].isin(cache_ids)]
        y = batchData[["Label"]].values
        X = batchData.iloc[:, 2:].values

        # Reshape input X into 2d array to express gray scaled image
        X = X.reshape(X.shape[0], 28, 28, 1)
        X = X / 255.0  # Scale input data between [0,1]

        # One hot encode targets y
        y_onehot = np.zeros((y.shape[0], 10))
        for i in range(y_onehot.shape[0]):
            y_onehot[i, y[i]] = 1

        return X, y_onehot


class cifar10Loader(DataManager):
    """
    cifar10Loader Documentation:
    --------------------------

    Purpose
    ----------
    Load data as needed from the cifar10 dataset.

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
        super().__init__("CIFAR10", bins, keep_bins)

        # For this data set, it is possible to read in all data at one
        self.data_df = pd.read_csv(self.data_path + "/raw_data/data_tab.csv")

    def image_gen(self, img_paths, img_size=None):
        """ Function to load images based on passed img_paths """
        # Iterate over all the image paths
        images = []
        for img_path in img_paths:
            # Load the image and mask, and normalize it to 0-1 range
            img = imread(img_path) / 255.
            # Resize the images
            if img_size != None:
                img = resize(img, img_size, preserve_range=True)
            # append the image
            images.append(img)
        return np.array(images)

    def getBatch(self, cache_ids: list):
        """
        Purpose
        _______
        Receives from algo class list of sample ids to get from memory or RAM

        Arguments
        __________
        cache_ids : list
            list contains ids of what data samples to send back

        Returns
        _______
        X : np.array
            Input values for model
        y : Target values for model

        """

        # Select data based off of passed cache_ids
        img_paths = [self.data_path + "/raw_data/images/" + str(id) + ".png" for id in cache_ids]

        X = self.image_gen(img_paths)  # shape (batch_size,32,32,3)

        batchData = self.data_df.loc[self.data_df['ID'].isin(cache_ids)]
        y = batchData[["y"]].values

        # One hot encode targets y
        y_onehot = np.zeros((y.shape[0], 10))
        for i in range(y_onehot.shape[0]):
            y_onehot[i, y[i]] = 1

        return X, y_onehot


class beeLoader(DataManager):
    """
    beeLoader Documentation:
    --------------------------

    Purpose
    ----------
    Load data as needed from the Bees dataset.

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
        super().__init__("Bees", bins, keep_bins)

        # For this data set, it is possible to read in all data at one
        self.data_df = pd.read_csv(self.data_path + "/raw_data/data_tab.csv")
        self.img_path = self.data_path + "/raw_data/images/"

    def getSample(self, id) -> (np.array, np.array):
        """Load image into np.array and make one-hot target array

        Parameters
        ----------
        sample : np.array
            an row from samples (np.array storing ids and class_ids
            of bee data)

        Returns
        -------
        img: np.array
            a np.array containing bee image data
        target: np.array
            a np.array as a one-hot vector expressing bee type
        """

        class_id = self.data_df[self.data_df["ID"] == id]
        class_id = class_id["class_id"].values

        img = Image.open(self.img_path + str(id) + ".png")
        img = img.resize((80, 80))
        img = np.array(img) / 255
        if img.shape != (80, 80, 3):
            img = img[:, :, :3]

        target = np.zeros(6)
        target[class_id] = 1
        return img, target

    def getBatchData(self, samples):
        """Load multiple images into np.array and make one-hot target arrays

        Parameters
        ----------
        samples : np.array
            multiple rows from samples (np.array storing ids and class_ids
            of bee data)

        Returns
        -------
        imgs: np.array
            a np.array containing multiple bee image data
        targets: np.array
            a np.array containing multiple one-hot vector expressing bee types
        """
        imgs, targets = [], []
        for i in range(len(samples)):
            X_sample, y_sample = self.getSample(samples[i])
            imgs.append(X_sample)
            targets.append(y_sample)

        imgs = np.concatenate(imgs, axis=0)
        imgs = imgs.reshape(len(samples), 80, 80, 3)
        targets = np.stack(targets, axis=0)

        return imgs, targets

    def getBatch(self, cache_ids: list):
        """
        Purpose
        _______
        Receives from algo class list of sample ids to get from memory or RAM

        Arguments
        __________
        cache_ids : list
            list contains ids of what data samples to send back

        Returns
        _______
        X : np.array
            Input values for model
        y : Target values for model

        """

        # Select data based off of passed cache_ids
        X, y = self.getBatchData(cache_ids)  # shape (batch_size,80,80,3)
        return X, y


class birdsLoader(DataManager):
    """
    birdsLoader Documentation:
    --------------------------

    Purpose
    ----------
    Load data as needed from the Bees dataset.

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
        super().__init__("Birds", bins, keep_bins)

        # For this data set, it is possible to read in all data at one
        self.data_df = pd.read_csv(self.data_path + "/raw_data/data_tab.csv")
        self.img_path = self.data_path + "/raw_data/images/"

    def getSample(self, id) -> (np.array, np.array):
        """Load image into np.array and make one-hot target array

        Parameters
        ----------
        sample : np.array
            an row from samples (np.array storing ids and class_ids
            of bee data)

        Returns
        -------
        img: np.array
            a np.array containing bee image data
        target: np.array
            a np.array as a one-hot vector expressing bee type
        """

        class_id = self.data_df[self.data_df["ID"] == id]
        class_id = class_id["class_id"].values

        img = Image.open(self.img_path + str(id) + ".jpg").convert('RGB')
        img = img.resize((200,200))
        img = np.array(img) / 255
        if img.shape != (200, 200, 3):
            print(self.img_path + str(id) + ".jpg")
            img = img[:, :, :3]

        #target = np.zeros(200)
        #target[class_id] = 1
        #return img, target
        return img, class_id

    def getBatchData(self, samples):
        """Load multiple images into np.array and make one-hot target arrays

        Parameters
        ----------
        samples : np.array
            multiple rows from samples (np.array storing ids and class_ids
            of bee data)

        Returns
        -------
        imgs: np.array
            a np.array containing multiple bee image data
        targets: np.array
            a np.array containing multiple one-hot vector expressing bee types
        """
        imgs, targets = [], []
        for i in range(len(samples)):
            X_sample, y_sample = self.getSample(samples[i])
            imgs.append(X_sample)
            targets.append(y_sample)

        imgs = np.concatenate(imgs, axis=0)
        imgs = imgs.reshape(len(samples), 200, 200, 3)
        targets = np.stack(targets, axis=0)

        return imgs, targets

    def getBatch(self, cache_ids: list):
        """
        Purpose
        _______
        Receives from algo class list of sample ids to get from memory or RAM

        Arguments
        __________
        cache_ids : list
            list contains ids of what data samples to send back

        Returns
        _______
        X : np.array
            Input values for model
        y : Target values for model

        """

        # Select data based off of passed cache_ids
        X, y = self.getBatchData(cache_ids)  # shape (batch_size,80,80,3)
        return X, y


class monkeyLoader(DataManager):
    """
    monkeyLoader Documentation:
    --------------------------

    Purpose
    ----------
    Load data as needed from the Bees dataset.

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
        super().__init__("Monkey", bins, keep_bins)

        # For this data set, it is possible to read in all data at one
        self.data_df = pd.read_csv(self.data_path + "/raw_data/data_tab.csv")
        self.img_path = self.data_path + "/raw_data/images/"

    def getSample(self, id) -> (np.array, np.array):
        """Load image into np.array and make one-hot target array

        Parameters
        ----------
        sample : np.array
            an row from samples (np.array storing ids and class_ids
            of bee data)

        Returns
        -------
        img: np.array
            a np.array containing bee image data
        target: np.array
            a np.array as a one-hot vector expressing bee type
        """

        class_id = self.data_df[self.data_df["ID"] == id]
        class_id = class_id["class_id"].values
        img = Image.open(self.img_path + str(id) + ".jpg").convert('RGB')
        img = np.array(img) / 255
        target = np.zeros(10)
        target[class_id] = 1
        return img, target

    def getBatchData(self, samples):
        """Load multiple images into np.array and make one-hot target arrays

        Parameters
        ----------
        samples : np.array
            multiple rows from samples (np.array storing ids and class_ids
            of bee data)

        Returns
        -------
        imgs: np.array
            a np.array containing multiple bee image data
        targets: np.array
            a np.array containing multiple one-hot vector expressing bee types
        """
        imgs, targets = [], []
        for i in range(len(samples)):
            X_sample, y_sample = self.getSample(samples[i])
            imgs.append(X_sample)
            targets.append(y_sample)

        imgs = np.concatenate(imgs, axis=0)
        imgs = imgs.reshape(len(samples), 120, 120, 3)
        targets = np.stack(targets, axis=0)

        return imgs, targets

    def getBatch(self, cache_ids: list):
        """
        Purpose
        _______
        Receives from algo class list of sample ids to get from memory or RAM

        Arguments
        __________
        cache_ids : list
            list contains ids of what data samples to send back

        Returns
        _______
        X : np.array
            Input values for model
        y : Target values for model

        """

        # Select data based off of passed cache_ids
        X, y = self.getBatchData(cache_ids)  # shape (batch_size,80,80,3)
        return X, y
