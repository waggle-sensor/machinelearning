import random
import numpy as np
import pandas as pd


#######################################################

class alAlgo:
    """
    Parent class that will be used for making new Active Learning algo classes.
    Currently, the class is very sparse. Will make adjustments as the project continues.
    """

    def __init__(self, algo_name = "NA"):
        self.algo_name = algo_name
        self.round = 0
        self.sample_log = {}

    def reset(self):
        self.round = 0
        self.sample_log = {}


#######################################################

"""

# Example of a child class of alAlgo.
# Use as template to make new Active Learning Algorithms

class child_alAlgo(alAlgo):

    def __init__(self):
        super().__init__()
        self.dummy = None

    def __call__(self, cache: list, n: int, **data) -> list:
        
        # Custom methodology to select batch
        batch = F(cache, a, b, c)

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        # Increment round
        self.round += 1

        return batch

"""


#######################################################

class leastConfidence(alAlgo):
    """
    Score samples by predictions through formula LC(x)=(1-P(y*|x))*(n/(n-1))
    """

    def __init__(self):
        super().__init__(algo_name="Least Confidence")
        self.sample_log = {}
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
    Randomly samples over a uniform distribution of passed cache of data ids.
    Use as a baseline to compare the performance of your active learning algorithms.

    """

    def __init__(self):
        super().__init__(algo_name="Passive")
        self.sample_log = {}
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
