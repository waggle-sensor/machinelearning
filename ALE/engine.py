from math import floor
from random import shuffle
from itertools import chain
from tqdm import trange
import matplotlib.pyplot as plt


#######################################################

def testCallEngine():
    print("Called test function from ~/engine.py")


class Engine():
    """
    Engine() Documentation:
    --------------------------

    Purpose
    ----------
    Control experiment for active learning. Schedules when to use the data loader, the model
    and the active learning algo to sample.

    Attributes
    ----------
    algoClass : alAlgo()
        used to decide what samples to take each round
    dataClass : DataManager()
        used to fetch and pre-process data for each round or initial training
    zooKeeper: zooKeeper()
        used to store tensorflow model and methods for training and testing the model.

    Methods
    -------
    trainBatch(self, cycles, batch_size):
        Used to call zooKeeper to peform training of tf model

    initialTrain(self, epochs, batch_size, val=True, plot=False:
        Used to train tf model from zooKeeper on initially provided training data

    valTest(self, batch_size):
        Used to check model's loss on validation data set

    plotLog(self, logs, xlabel, ylabel, title, labels):
        Plots tracked metrics. To modify plot, do so here.

    run(self, rounds, cycles, batch_size, val=True, plot=False):
        Used to track and update cache of each active learning round. Controls runCycle()
        method which is what calls for data, model update, and update cache. If you wanted to
        have very very very precise control of what happens between training rounds, modify code here.
        Warning it is very intricate and could easily break the code, do so at your own risk.

    """

    def __init__(self, algoClass, dataClass, zooKeeper, sample_size):
        self.intialTrain_metric_log = {}
        self.intialVal_metric_log = {}
        self.train_metric_log = {}
        self.val_metric_log = {}

        # Declare passed objects
        self.algoClass = algoClass  # Class for Active Learning algo with method to select batches
        self.dataClass = dataClass  # Data class that stores cache df and has method to load in data based off of algoClass batch
        self.modelManager = zooKeeper  # Model class that stores model and has train and predict methods

        self.round = 0  # Keep track of what round of training the system is on
        self.sample_size = sample_size

        self.multipleBins = False
        if self.dataClass.bins > 1:
            self.multipleBins = True

    def trainBatch(self, cycles, batch_size):
        total_loss = []
        for i in range(cycles):
            for batch in trange(floor(len(self.dataClass.train_cache) / batch_size)):
                batch_ids = self.dataClass.train_cache[batch_size * batch:batch_size * (batch + 1)]
                X, y = self.dataClass.getBatch(batch_ids)
                loss = self.modelManager.modelObject.trainBatch(X, y)
                total_loss.append(loss)
        return total_loss

    def valTest(self, batch_size):
        total_loss = []
        for batch in trange(floor(len(self.dataClass.val_cache) / batch_size)):
            batch_ids = self.dataClass.val_cache[batch_size * batch:batch_size * (batch + 1)]
            X, y = self.dataClass.getBatch(batch_ids)
            loss = self.modelManager.modelObject.eval(X, y)
            total_loss.append(loss)
        return total_loss

    def plotLog(self, logs, xlabel, ylabel, title, labels):
        fig, ax = plt.subplots()
        for i, log in enumerate(logs):
            ax.scatter(list(range(len(log))), list(log.values()), label=labels[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        ax.legend()
        ax.grid(True)

        plt.show()

    def runCycle(self, batch_size, cycles):
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
        # 2. Train and log
        # ------------------------------------
        total_loss = self.trainBatch(cycles, batch_size)
        total_loss = list(chain(*total_loss))
        avg_loss = sum(total_loss) / len(total_loss)
        print("Training loss: {}".format(self.round, avg_loss))
        self.train_metric_log["Round_" + str(self.round)] = avg_loss.numpy()

    def run(self, rounds, cycles, batch_size, val=True, plot=False):
        """
        Purpose: Calls runCycle through for loop to perform active learning training
        :param rounds: int that determines number of active learning training rounds
        :return: None
        """
        for i in range(rounds):
            print("Round: {}".format(i))
            self.runCycle(batch_size, cycles)

            if val == True:
                total_loss = self.valTest(batch_size)
                total_loss = list(chain(*total_loss))
                avg_loss = sum(total_loss) / len(total_loss)
                print("{}. Val loss: {}".format(self.round, avg_loss))
                self.val_metric_log["Round_" + str(self.round)] = avg_loss.numpy()

            self.round += 1

        if plot == True:
            xlabel = "Round"
            ylabel = "Error"
            title = "Round vs Error on Active Learning"
            if val == True:
                labels = ["Train", "Val"]
                self.plotLog([self.train_metric_log, self.val_metric_log], xlabel, ylabel, title, labels)
            else:
                self.plotLog([self.train_metric_log], xlabel, ylabel, title, "Train")
            input('press return to continue')

    def initialTrain(self, epochs, batch_size, val=True, plot=False):
        for epoch in range(epochs):
            total_loss = self.trainBatch(1, batch_size)
            total_loss = list(chain(*total_loss))
            avg_loss = sum(total_loss) / len(total_loss)
            print("{}. training average loss: {}".format(epoch, avg_loss))
            self.intialTrain_metric_log["Epoch_" + str(epoch)] = avg_loss.numpy()

            if val == True:
                total_loss = self.valTest(batch_size)
                total_loss = list(chain(*total_loss))
                avg_loss = sum(total_loss) / len(total_loss)
                print("{}. Val loss: {}".format(self.round, avg_loss))
                self.intialVal_metric_log["Epoch_" + str(epoch)] = avg_loss.numpy()

        print('Finished initial training of model.')

        if plot == True:
            xlabel = "Epoch"
            ylabel = "Error"
            title = "Epoch vs Error on Train"
            if val == True:
                labels = ["Train", "Val"]
                self.plotLog([self.intialTrain_metric_log, self.intialVal_metric_log], xlabel, ylabel, title, labels)
            else:
                self.plotLog([self.intialTrain_metric_log], xlabel, ylabel, title, "Train")
            input('press return to continue')
