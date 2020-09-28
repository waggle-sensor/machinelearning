# Import modules
import abc
import random
import numpy as np
import pandas as pd


#######################################################

class alAlgo(metaclass=abc.ABCMeta):
    """
    alAlgo() Documentation:
    --------------------------

    Purpose
    ----------
    Parent class that will be used for making new Active Learning algo classes.
    Currently, the class is very sparse. Will make adjustments as the project continues.

    Attributes
    ----------
    algo_name : str
        used to keep track of name of algo in engine.log

    sample_log : dict
        tracks what samples are chosen each round, places sample ids in list within dict

    round : int
        tracks what round algo is on

    predict_to_sample : bool
        bool that determines whether or not the algo needs the predictions of the model to choose which samples to label

    Methods
    -------
    @classmethod
    __subclasshook__(cls, subclass):
        Used to check if custom child class of alAlgo is properly made

    reset(self):
        set round=0 and sample_log={}

    @abc.abstractmethod
    __call__(self, cache: list, n: int, yh):
        Empty function that is required to be declared in custom child class. Allows for algo
        to be called to pick which samples to return based on algo criteria.
    """

    def __init__(self, algo_name="NA"):
        self.algo_name = algo_name
        self.round = 0
        self.sample_log = {}
        self.predict_to_sample = False

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__call__') and
                callable(subclass.__call__) or
                NotImplemented)

    def reset(self):
        self.round = 0
        self.sample_log = {}

    @abc.abstractmethod
    def __call__(self, cache: list, n: int, yh):
        """ Selects which samples to get labels for """
        raise NotImplementedError


#######################################################

class leastConfidence(alAlgo):
    """
    leastConfidence(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.
    Score samples by predictions through formula LC(x)=(1-P(y*|x))*(n/(n-1))

    Attributes
    ----------
    predict_to_sample : bool
        Determines if algo needs models prediction on cache to determine what samples from the cache to return

    Methods
    -------
    @abc.abstractmethod
    __call__(self, cache: list, n: int, yh):
        Empty function that is required to be declared in custom child class. Allows for algo
        to be called to pick which samples to return based on algo criteria.
    """

    def __init__(self):
        super().__init__(algo_name="Least Confidence")
        self.predict_to_sample = True

    def __call__(self, cache: list, n: int, yh) -> list:

        # Check if embedded cache, then cache is available for the round
        if any(isinstance(i, list) for i in cache):
            try:
                cache = cache[self.round]
            except:
                raise ValueError("Active Learning Algo has iterated through each round\'s unlabled cache.")

        # Check if sample size is to large for cache
        if len(cache) < n:
            raise ValueError("Sample size n is larger than length of round's cache")

        # Calculate LC(x) values
        yh_vals = yh.iloc[:, 1:].values
        LC_vals = []
        for i in range(yh_vals.shape[0]):
            sample = yh_vals[i, :]
            lc = (1 - np.amax(sample)) * (yh_vals.shape[1] / (yh_vals.shape[1] - 1))
            LC_vals.append((lc))

        target_col_names = ["y" + str(i) for i in range(yh_vals.shape[1])]
        yh_col_names = ["LC", "ID"] + target_col_names
        yh = pd.concat([pd.DataFrame(LC_vals), yh], axis=1)
        yh.columns = yh_col_names

        # Get ids of n largest LC vals
        n_largest = yh.nlargest(n, 'LC')
        batch = n_largest["ID"].to_list()

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("\n")
        print("Round {} selected samples: {}".format(self.round, batch))
        print("\n")

        # Increment round
        self.round += 1

        return batch


#######################################################

class uniformSample(alAlgo):
    """
    uniformSample(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.
    Randomly samples over a uniform distribution of passed cache of data ids.
    Use as a baseline to compare the performance of your active learning algorithms.

    Attributes
    ----------
    predict_to_sample : bool
        Determines if algo needs models prediction on cache to determine what samples from the cache to return

    Methods
    -------
    @abc.abstractmethod
    __call__(self, cache: list, n: int, yh):
        Empty function that is required to be declared in custom child class. Allows for algo
        to be called to pick which samples to return based on algo criteria.
    """

    def __init__(self):
        super().__init__(algo_name="Passive")
        self.predict_to_sample = False

    def __call__(self, cache: list, n: int, yh) -> list:
        # Check if embedded cache, then cache is available for the round
        if any(isinstance(i, list) for i in cache):
            try:
                cache = cache[self.round]
            except:
                raise ValueError("Active Learning Algo has iterated through each round\'s unlabled cache.")

        # Check if sample size is to large for cache
        if len(cache) < n:
            raise ValueError("Sample size n is larger than length of round's cache")

        # Select from uniform distributions data ID's from given cache
        idx = random.sample(range(0, len(cache)), n)
        batch = [cache[i] for i in idx]

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("\n")
        print("Round {} selected samples: {}".format(self.round, idx))
        print("\n")

        # Increment round
        self.round += 1

        return batch


#######################################################


class ratioConfidence(alAlgo):
    """
    ratioConfidence(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.
    Score samples by predictions through formula theta(x)=P(y_1/x)/P(y_2/x)

    Attributes
    ----------
    predict_to_sample : bool
        Determines if algo needs models prediction on cache to determine what samples from the cache to return

    Methods
    -------
    @abc.abstractmethod
    __call__(self, cache: list, n: int, yh):
        Empty function that is required to be declared in custom child class. Allows for algo
        to be called to pick which samples to return based on algo criteria.
    """

    def __init__(self):
        super().__init__(algo_name="Ratio Confidence")
        self.predict_to_sample = True

    def __call__(self, cache: list, n: int, yh) -> list:

        # Check if embedded cache, then cache is available for the round
        if any(isinstance(i, list) for i in cache):
            try:
                cache = cache[self.round]
            except:
                raise ValueError("Active Learning Algo has iterated through each round\'s unlabled cache.")

        # Check if sample size is to large for cache
        if len(cache) < n:
            raise ValueError("Sample size n is larger than length of round's cache")

        # Calculate LC(x) values
        yh_vals = yh.iloc[:, 1:].values
        RC_vals = []
        for i in range(yh_vals.shape[0]):
            sample = yh_vals[i, :]
            sample[::-1].sort()
            y1, y2 = sample[0], sample[1]
            if y2 == 0:
                RC_vals.append(100)
            else:
                RC_vals.append(y1/y2)

        target_col_names = ["y" + str(i) for i in range(yh_vals.shape[1])]
        yh_col_names = ["RC", "ID"] + target_col_names
        yh = pd.concat([pd.DataFrame(RC_vals), yh], axis=1)
        yh.columns = yh_col_names

        # Get ids of n largest LC vals
        n_smallest = yh.nsmallest(n, 'RC')
        batch = n_smallest["ID"].to_list()

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("\n")
        print("Round {} selected samples: {}".format(self.round, batch))
        print("\n")

        # Increment round
        self.round += 1

        return batch
