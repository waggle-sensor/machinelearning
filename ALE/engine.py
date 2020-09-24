import numpy as np
import pandas as pd
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

        self.log = self.getLog()
        print(self.log)

    def getLog(self):
        keys = ["round", self.dataClass.dataset_name, self.algoClass.algo_name, self.modelManager.modelName,
                "time_round_seconds", "batch_size", "n_train", "n_val"]

        # Add col names for unlabeled caches
        if self.dataClass.bins == 1:
            temp_col_names = ["u_cache"]
        else:
            temp_col_names = ["u_cache_" + str(i) for i in range(self.dataClass.bins)]
        keys.extend(temp_col_names)

        # Add col names for loss metric
        temp_col_names = ["loss_train_" + self.modelManager.modelObject.loss.__name__,
                          "loss_val_" + self.modelManager.modelObject.loss.__name__]
        keys.extend(temp_col_names)

        # Add col names for extra metrics
        temp_col_names = ["train_" + metric.name for i, metric in enumerate(self.modelManager.metrics)]
        keys.extend(temp_col_names)
        temp_col_names = ["val_" + metric.name for i, metric in enumerate(self.modelManager.metrics)]
        keys.extend(temp_col_names)

        log = {k: [] for k in keys}
        return log

    # TODO: update log function, this is going to be a pain [Plan out before coding]
    def updateLog(self):
        pass

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
        total_scores = []
        for batch in trange(floor(len(self.dataClass.val_cache) / batch_size)):
            batch_ids = self.dataClass.val_cache[batch_size * batch:batch_size * (batch + 1)]
            X, y = self.dataClass.getBatch(batch_ids)
            loss, scores, _ = self.modelManager.modelObject.eval(X, y)
            total_loss.append(loss)
            total_scores.append(scores)
        return total_loss, total_scores

    def evalCache(self, cache, batch_size):
        predictions = []
        for batch in trange(floor(len(cache) / batch_size)):
            batch_ids = cache[batch_size * batch:batch_size * (batch + 1)]
            X, y = self.dataClass.getBatch(batch_ids)
            _, _, yh = self.modelManager.modelObject.eval(X, y)
            predictions.append(yh)
        return np.concatenate(predictions, axis=0)

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

        if self.dataClass.bins > 1:
            if self.round >= self.dataClass.bins:
                raise ValueError(
                    'Error: Engine has been called to run rounds more times than number of unlabeled caches!')
            cache_df = self.dataClass.unlabeled_cache[self.round]
        else:
            cache_df = self.dataClass.unlabeled_cache

        # Get subset of cache ids based off of active learning algo
        if self.algoClass.predict_to_sample == False:
            ids = self.algoClass(cache_df, self.sample_size)
        else:
            # cache_df
            yh = self.evalCache(cache_df, batch_size)
            yh = pd.concat([pd.DataFrame(cache_df[:len(yh)]), pd.DataFrame(yh)], axis=1)
            ids = self.algoClass(cache_df, self.sample_size, yh)

        # Manage new labels within caches
        self.dataClass.train_cache.extend(ids)  # Add new ids to train cache

        shuffle(self.dataClass.train_cache)  # Shuffle new dataClass.train_cache

        # Remove new ids from unlabeled caches: (two cases) one or multiple unlabeled caches
        if self.dataClass.keep_bins == True and self.dataClass.bins > 1:
            for i in range(self.round, self.dataClass.bins):
                [self.dataClass.unlabeled_cache[i].remove(id) for id in ids]
        elif self.dataClass.bins == 1:
            [self.dataClass.unlabeled_cache.remove(id) for id in ids]

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

        print("\n" + "Active Learning algorithm start:")

        for i in range(rounds):
            print("Round: {}".format(i))
            self.runCycle(batch_size, cycles)

            if val == True:
                total_loss, total_scores = self.valTest(batch_size)
                total_loss = list(chain(*total_loss))
                avg_loss = sum(total_loss) / len(total_loss)
                print("{}. Val loss: {}".format(self.round, avg_loss))
                self.val_metric_log["Round_" + str(self.round)] = avg_loss.numpy()

                keys = list(total_scores[0].keys())
                for i, k in enumerate(keys):
                    samples = []
                    for j, sample in enumerate(total_scores):
                        samples.append((sample[k]))
                    samples = np.mean(samples)
                    print("{}: {}".format(self.modelManager.modelObject.metrics[i].name, samples))

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
            input('press return to continue' + "\n")

        print("\n" + "Active Learning algorithm done.")

    def initialTrain(self, epochs, batch_size, val=True, plot=False):
        print("\n" + "Training model on training cache:")
        for epoch in range(epochs):
            total_loss = self.trainBatch(1, batch_size)
            total_loss = list(chain(*total_loss))
            avg_loss = sum(total_loss) / len(total_loss)
            print("{}. training average loss: {}".format(epoch, avg_loss))
            self.intialTrain_metric_log["Epoch_" + str(epoch)] = avg_loss.numpy()

            if val == True:
                total_loss, total_scores = self.valTest(batch_size)

                total_loss = list(chain(*total_loss))
                avg_loss = sum(total_loss) / len(total_loss)
                print("{}. Val loss: {}".format(self.round, avg_loss))
                self.intialVal_metric_log["Epoch_" + str(epoch)] = avg_loss.numpy()

                keys = list(total_scores[0].keys())
                for i, k in enumerate(keys):
                    samples = []
                    for j, sample in enumerate(total_scores):
                        samples.append((sample[k]))
                    samples = np.mean(samples)
                    print("{}: {}".format(self.modelManager.modelObject.metrics[i].name, samples))

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
