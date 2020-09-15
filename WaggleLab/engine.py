from math import floor
from random import shuffle
from tqdm import trange


def testCallEngine():
    print("Called test function from ~/engine.py")


class Engine():
    def __init__(self, algoClass, dataClass, zooKeeper, sample_size):
        # Declare passed objects
        self.algoClass = algoClass  # Class for Active Learning algo with method to select batches
        self.dataClass = dataClass  # Data class that stores cache df and has method to load in data based off of algoClass batch
        self.modelManager = zooKeeper  # Model class that stores model and has train and predict methods

        self.round = 0  # Keep track of what round of training the system is on
        self.sample_size = sample_size

        self.multipleBins = False
        if self.dataClass.bins > 1:
            self.multipleBins = True

    def runCycle(self, batch_size):
        # ------------------------------------
        # 1. Get data and update caches
        # ------------------------------------

        # Get round unlabeled cache ids
        print("Active Learning training round: {}".format(self.round))

        if self.multipleBins == True:
            if self.round >= self.dataClass.bins:
                raise ValueError(
                    'Error: Engine has been called to run rounds more times than number of unlabeled caches!')
            cache_df = self.dataClass.unlabeled_cache[self.round]
        else:
            cache_df = self.dataClass.unlabeled_cache

        # Get subset of cache ids based off of active learning algo
        ids = self.algoClass(cache_df, self.sample_size)

        # Manage new labels within caches
        self.dataClass.train_cache.extend(ids)  # Add new ids to train cache

        shuffle(self.dataClass.train_cache)  # Shuffle new dataClass.train_cache
        if self.dataClass.keep_bins == True:  # Remove new ids from unlabeled caches: (two cases) one or multiple unlabled caches
            for i in range(self.round, self.dataClass.bins):
                [self.dataClass.unlabeled_cache[i].remove(id) for id in ids]
        else:
            self.dataClass.unlabeled_cache.remove(ids)

        # ------------------------------------
        # 2. Train and eval validation on batch
        # ------------------------------------
        # TODO: make train and eval routines
        for batch in trange(floor(len(self.dataClass.train_cache) / batch_size)):
            batch_ids = self.dataClass.train_cache[batch_size * batch:batch_size * (batch + 1)]
            X, y = self.dataClass.getBatch(batch_ids)
            # TODO: Left off here, pass X and y to train batch function in modelManager !!!

        # ------------------------------------
        # 3. Log results
        # ------------------------------------
        # TODO: log results routines

        # End round by
        self.round += 1

    def run(self, cycles, batch_size):
        """
        Purpose: Calls runCycle through for loop to perform active learning training
        :param rounds: int that determines number of active learning training rounds
        :return: None
        """
        for i in range(cycles):
            self.runCycle(batch_size)
