# Import modules
import abc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import floor
from itertools import chain

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential


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

class marginConfidence(alAlgo):
    """
    marginConfidence(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.
    Score samples by predictions through formula MC(x)=(1-(P(y1*|x)-P(y2*|x)))

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
        super().__init__(algo_name="Margin Confidence")
        self.predict_to_sample = True
        self.feature_set = False

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

        # Calculate MC(x) values
        yh_vals = yh.iloc[:, 1:].values
        MC_vals = []
        for i in range(yh_vals.shape[0]):
            sample = yh_vals[i, :]
            sample[::-1].sort()
            y1, y2 = sample[0], sample[1]
            mc_val = 1-(y1-y2)
            MC_vals.append(mc_val)

        target_col_names = ["y" + str(i) for i in range(yh_vals.shape[1])]
        yh_col_names = ["MC", "ID"] + target_col_names
        yh = pd.concat([pd.DataFrame(MC_vals), yh], axis=1)
        yh.columns = yh_col_names

        # Get ids of n largest LC vals
        n_largest = yh.nlargest(n, 'MC')
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
        self.feature_set = False

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
        self.feature_set = False

    def __call__(self, cache: list, n: int, yh=None) -> list:
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

        print("Selected samples: ")
        print(idx)
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
        self.feature_set = False

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

        # Calculate RC(x) values
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


#######################################################


class entropy(alAlgo):
    """
    ratioConfidence(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.
    Score samples by predictions through formula ent(x)= -sum(P(Y|X)log_{2}P(Y|X))/log_{2}

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
        self.feature_set = False

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

        # Calculate ent(x) values
        yh_vals = yh.iloc[:, 1:].values
        ent_vals = []
        for i in range(yh_vals.shape[0]):
            sample = yh_vals[i, :]
            log_probs = sample * np.log2(sample)  # multiply each proba by its base 2 log
            raw_entropy = 0 - np.sum(log_probs)
            normalized_entropy = raw_entropy / np.log2(len(sample))
            ent_vals.append(normalized_entropy)

        target_col_names = ["y" + str(i) for i in range(yh_vals.shape[1])]
        yh_col_names = ["ENT", "ID"] + target_col_names
        yh = pd.concat([pd.DataFrame(ent_vals), yh], axis=1)
        yh.columns = yh_col_names

        # Get ids of n largest LC vals
        n_largest = yh.nlargest(n, 'ENT')
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


class DAL(alAlgo):
    """
    DAL(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.
    Score samples by predictions through formula ent(x)= -sum(P(Y|X)log_{2}P(Y|X))/log_{2}

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

    def __init__(self,input_dim=None):
        super().__init__(algo_name="DAL")
        self.predict_to_sample = False
        self.feature_set = True

        if input_dim == None:
            raise ValueError("Must pass input dim as int to use DAL")
        self.input_dim = input_dim
        self.model = self.getBinaryClassifier()

        self.opt = tf.keras.optimizers.Adam(lr=0.0001)
        self.loss = tf.keras.losses.categorical_crossentropy

    def getBinaryClassifier(self):
        model = Sequential(name="Binary Classifier")
        model.add(Dense(128, activation='elu', input_dim=self.input_dim))
        model.add(Dropout(.1))
        model.add(Dense(2, activation='softmax'))
        return model

    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(self.model(inputs, training=True), targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def trainBatch(self, inputs, targets) -> float:
        """ Calculates loss and gradients for batch of data and applies update to weights """
        loss_value, grads = self.grad(inputs, targets)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    def predict(self, inputs):
        """ Used for predicting with model but does not have labels """
        yh = self.model(inputs)
        return yh

    def trainBinaryClassifier(self,dataset,batch_size):
        remainder_samples = dataset.shape[0] % batch_size # Calculate number of remainder samples from batches
        total_loss= []

        # Run batches
        print("Training DAL Binary Classifier")
        for i in tqdm(range(50)):
            for batch in range(floor(dataset.shape[0] / batch_size)):
                data = dataset[batch_size * batch:batch_size * (batch + 1),:]
                X, y = data[:,:-2], data[:,-2:]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            # Run remainders
            if remainder_samples > 0:
                data = dataset[(-1)*remainder_samples:,:]
                X, y = data[:,:-2], data[:,-2:]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            total_loss = list(chain(*total_loss))
            val_avg_loss = sum(total_loss) / len(total_loss)
            total_loss = []
        print("DAL binary classifier loss: {}".format(val_avg_loss))


    def inferBinaryClassifier(self,inputs):
        yh = self.model(inputs)
        return yh

    def resetBinayClassifier(self):
        pass

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
        yh_vals = yh.iloc[:, 1].values
        yh_col_names = ["yh", "ID"]
        yh = pd.concat([pd.DataFrame(yh_vals), pd.DataFrame(cache)], axis=1)
        yh.columns = yh_col_names

        # Get ids of n largest LC vals
        n_largest = yh.nlargest(n, 'yh')
        batch = n_largest["ID"].to_list()

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("\n")
        print("Round {} selected samples: {}".format(self.round, batch))
        print("\n")

        # Increment round
        self.round += 1

        return batch



