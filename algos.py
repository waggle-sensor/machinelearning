import random


#######################################################

class alAlgo:
    """
    Parent class that will be used for making new Active Learning algo classes.
    Currently, the class is very sparse. Will make adjustments as the project continues.
    """

    def __init__(self):
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

class uniformSample(alAlgo):
    """
    Randomly samples over a uniform distribution of passed cache of data ids.
    Use as a baseline to compare the performance of your active learning algorithms.

    """

    def __init__(self):
        super().__init__()
        self.sample_log = {}

    def __call__(self, cache: list, n: int) -> list:
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
