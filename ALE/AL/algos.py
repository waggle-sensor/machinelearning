# Import modules
import abc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import floor
from itertools import chain

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from typeguard import typechecked

from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans


import matplotlib.pyplot as plt

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
        self.single_output = False

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
            mc_val = 1 - (y1 - y2)
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
        self.single_output = False

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
        self.single_output = False

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
        self.single_output = False

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
                RC_vals.append(y1 / y2)

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
        self.single_output = False

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

    def __init__(self, input_dim=None):
        super().__init__(algo_name="DAL")
        self.predict_to_sample = False
        self.feature_set = True
        self.single_output = False

        if input_dim == None:
            raise ValueError("Must pass input dim as int to use DAL")
        self.input_dim = input_dim
        self.model = self.getBinaryClassifier()

        self.opt = tf.keras.optimizers.Adam(lr=0.0001)
        self.loss = tf.keras.losses.categorical_crossentropy
        # self.loss = tf.keras.losses.kl_divergence

    def getBinaryClassifier(self):
        model = Sequential(name="BC")
        model.add(Dense(120, activation='relu', input_dim=self.input_dim))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        return model

    @tf.function
    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(self.model(inputs, training=True), targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    @tf.function
    def trainBatch(self, inputs, targets) -> float:
        """ Calculates loss and gradients for batch of data and applies update to weights """
        loss_value, grads = self.grad(inputs, targets)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    @tf.function
    def predict(self, inputs):
        """ Used for predicting with model but does not have labels """
        yh = self.model(inputs)
        return yh

    def trainBinaryClassifier(self, dataset, batch_size):
        remainder_samples = dataset.shape[0] % batch_size  # Calculate number of remainder samples from batches
        total_loss = []

        # Test resetting binary classifier each round
        #self.model = self.getBinaryClassifier()

        # Run batches
        print("Training DAL Binary Classifier")
        for i in tqdm(range(100)):
            for batch in range(floor(dataset.shape[0] / batch_size)):
                data = dataset[batch_size * batch:batch_size * (batch + 1), :]
                X, y = data[:, :-2], data[:, -2:]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            # Run remainders
            if remainder_samples > 0:
                data = dataset[(-1) * remainder_samples:, :]
                X, y = data[:, :-2], data[:, -2:]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            total_loss = list(chain(*total_loss))
            val_avg_loss = sum(total_loss) / len(total_loss)
            total_loss = []
        print("DAL binary classifier loss: {}".format(val_avg_loss))

    def inferBinaryClassifier(self, inputs):
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


#######################################################


class DALratio(alAlgo):
    """
    DAL(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.

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

    def __init__(self, input_dim=None):
        super().__init__(algo_name="DALratio")
        self.predict_to_sample = False
        self.feature_set = True
        self.single_output = False

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

    def trainBinaryClassifier(self, dataset, batch_size):
        remainder_samples = dataset.shape[0] % batch_size  # Calculate number of remainder samples from batches
        total_loss = []

        # Run batches
        print("Training DAL Binary Classifier")
        for i in tqdm(range(50)):
            for batch in range(floor(dataset.shape[0] / batch_size)):
                data = dataset[batch_size * batch:batch_size * (batch + 1), :]
                X, y = data[:, :-2], data[:, -2:]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            # Run remainders
            if remainder_samples > 0:
                data = dataset[(-1) * remainder_samples:, :]
                X, y = data[:, :-2], data[:, -2:]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            total_loss = list(chain(*total_loss))
            val_avg_loss = sum(total_loss) / len(total_loss)
            total_loss = []
        print("DAL binary classifier loss: {}".format(val_avg_loss))

    def inferBinaryClassifier(self, inputs):
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
        yh1_vals = yh.iloc[:, 0].values
        yh2_vals = yh.iloc[:, 1].values
        yh_vals = np.absolute(yh1_vals - yh2_vals)

        yh_col_names = ["yh", "ID"]
        yh = pd.concat([pd.DataFrame(yh_vals), pd.DataFrame(cache)], axis=1)
        yh.columns = yh_col_names

        # Get ids of n largest LC vals
        n_largest = yh.nsmallest(n, 'yh')
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


class OC(alAlgo):
    """
    OC(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.

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

    def __init__(self, input_dim=None):
        super().__init__(algo_name="OC")
        self.predict_to_sample = False
        self.feature_set = True
        self.single_output = False

        if input_dim == None:
            raise ValueError("Must pass input dim as int to use DAL")
        self.input_dim = input_dim
        self.k = 500
        self.opt = tf.keras.optimizers.RMSprop(lr=0.0001)
        # self.loss = tf.keras.metrics.Mean()
        self.loss = tf.keras.losses.categorical_crossentropy
        self.model = self.getBinaryClassifier()

    def getBinaryClassifier(self):
        inputs = tf.keras.Input((self.input_dim,))
        out = tf.keras.layers.Dense(self.k, activation='relu', use_bias=False, name='certificates')(inputs)
        model = tf.keras.models.Model(inputs=[inputs], outputs=out, name='ONC')
        return model

    def grad(self, inputs):
        with tf.GradientTape() as tape:
            y_hat = self.model(inputs, training=True)

            # compute the loss
            # error = tf.math.reduce_mean(tf.math.square(y_hat))
            error = self.loss(y_hat, tf.zeros(y_hat.shape) + .0001)
            error = tf.cast(error, dtype=tf.dtypes.float64)

            W = self.model.layers[1].get_weights()[0]  # Equation 4.
            W = tf.linalg.matmul(tf.transpose(W), W)

            W = tf.cast(W, dtype=tf.dtypes.float64)

            penalty = tf.math.square(W - tf.eye(self.k, dtype=tf.dtypes.float64)) * 10
            penalty = tf.math.reduce_mean(penalty)

            error = error + penalty

            loss_value = self.loss(y_hat, tf.zeros(y_hat.shape) + .0001)

        return loss_value, tape.gradient(error, self.model.trainable_variables)

    def trainBatch(self, inputs) -> float:
        """ Calculates loss and gradients for batch of data and applies update to weights """
        loss_value, grads = self.grad(inputs)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        # self.model.layers[1].get_weights()[0] = tf.clip_by_value(self.model.layers[1].get_weights()[0],-.01,.01,)
        return loss_value

    def predict(self, inputs):
        """ Used for predicting with model but does not have labels """
        yh = self.model(inputs)
        return yh

    def trainBinaryClassifier(self, dataset, batch_size):
        remainder_samples = dataset.shape[0] % batch_size  # Calculate number of remainder samples from batches
        total_loss = []

        # Run batches
        print("Training DALOC")
        for i in tqdm(range(300)):
            for batch in range(floor(dataset.shape[0] / batch_size)):
                X = dataset[batch_size * batch:batch_size * (batch + 1), :]
                loss = self.trainBatch(X)
                total_loss.append(loss)

            # Run remainders
            if remainder_samples > 0:
                X = dataset[(-1) * remainder_samples:, :]
                loss = self.trainBatch(X)
                total_loss.append(loss)

        # val_avg_loss = sum(total_loss) / len(total_loss)
        val_avg_loss = 0
        print("DAL binary classifier loss: {}".format(val_avg_loss))

    def inferBinaryClassifier(self, inputs):
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

        # Calculate OC(x) values
        yh_vals = yh.values

        # Calculate epistemic uncertainty
        scores = tf.math.reduce_mean(tf.math.square(yh_vals), axis=1).numpy()

        yh_col_names = ["yh", "ID"]
        yh = pd.concat([pd.DataFrame(scores), pd.DataFrame(cache)], axis=1)
        yh.columns = yh_col_names

        # Get ids
        yh = yh.sort_values(by=['yh'])
        # median_index = yh[yh["yh"] == yh["yh"].quantile(.95, interpolation='lower')]
        # median_index = median_index.index.values[0]

        # n_largest = list(random.sample(range(median_index, yh.shape[0]), n))
        # n_largest = yh.iloc[n_largest,:]
        n_largest = yh.iloc[-n:, :]
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


class AADA(alAlgo):
    """
    AADA(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.

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

    def __init__(self, input_dim=None):
        super().__init__(algo_name="AADA")
        self.predict_to_sample = False
        self.feature_set = True
        self.single_output = True

        if input_dim == None:
            raise ValueError("Must pass input dim as int to use AADA")
        self.input_dim = input_dim
        self.model = self.getBinaryClassifier()

        self.opt = tf.keras.optimizers.RMSprop(lr=0.00005)
        # self.loss = tf.keras.losses.categorical_crossentropy
        self.loss = tf.keras.losses.mean_absolute_error

    def getBinaryClassifier(self):
        model = Sequential(name="AADA")
        model.add(Dense(128, activation='elu', input_dim=self.input_dim))
        # model.add(Dropout(.05))
        model.add(Dense(1, activation='linear'))
        # model.add(Dense(1, activation='linear'))
        return model

    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            yh = self.model(inputs, training=True)
            loss_value = tf.math.reduce_mean(targets * yh)

            # x_source = inputs[0]
            # x_target = inputs[1]

            # yh_source = self.model(x_source, training=True)
            # yh_target = self.model(x_target, training=True)

            # loss_value = tf.math.reduce_mean(yh_source)
            # loss_value = loss_value - tf.math.reduce_mean(yh_target)

            # loss_value = tf.math.reduce_mean(tf.math.log(yh_source+.01))
            # loss_value = -(loss_value + tf.math.reduce_mean(tf.math.log(1.01 - yh_target)))

        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def trainBatch(self, inputs, targets) -> float:
        """ Calculates loss and gradients for batch of data and applies update to weights """
        loss_value, grads = self.grad(inputs, targets)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    @tf.function
    def predict(self, inputs):
        """ Used for predicting with model but does not have labels """
        yh = self.model(inputs)
        return yh

    def trainBinaryClassifier(self, dataset, batch_size):
        # source_data = dataset[0]
        # target_data = dataset[1]
        # remainder_samples = target_data.shape[0] % batch_size
        remainder_samples = dataset.shape[0] % batch_size  # Calculate number of remainder samples from batches
        total_loss = []

        # Run batches
        print("Training AADA Classifier")
        for i in tqdm(range(100)):
            for batch in range(floor(dataset.shape[0] / batch_size)):
                data = dataset[batch_size * batch:batch_size * (batch + 1), :]
                X, y = data[:, :-1], data[:, -1]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            # Run remainders
            if remainder_samples > 0:
                data = dataset[(-1) * remainder_samples:, :]
                X, y = data[:, :-1], data[:, -1]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            np.random.shuffle(dataset)

            # total_loss = list(chain(*total_loss))
            # val_avg_loss = sum(total_loss) / len(total_loss)
            total_loss = []
        print("DAL binary classifier loss: {}".format(0))

    def inferBinaryClassifier(self, inputs):
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
        yh_vals = yh.values
        # yh_vals = (1-yh_vals)/yh_vals
        yh_col_names = ["yh", "ID"]
        yh = pd.concat([pd.DataFrame(yh_vals), pd.DataFrame(cache)], axis=1)
        yh.columns = yh_col_names

        print(yh_vals)

        # Get ids of n largest LC vals
        n_largest = yh.nlargest(n, 'yh')
        # n_largest = yh.nsmallest(n, 'yh')
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

class DALOC(alAlgo):
    """
    DALOC(alAlgo) Documentation:
    --------------------------

    Purpose
    DALOC implementation trains a binary classifier to discern between unlabeled and labeled data.
    OC's are also trained on the labeled data. The binary classifier takes in all of the unlabeled data
    and then outputs softmax scores for uncertainty. I then select the top 90th quantile of values and
    from there select the top 'n' values based on OC scores.

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

    def __init__(self, input_dim=None):
        super().__init__(algo_name="DALOC")
        self.predict_to_sample = False
        self.feature_set = True
        self.single_output = False

        if input_dim == None:
            raise ValueError("Must pass input dim as int to use DAL")
        self.input_dim = input_dim
        self.k = 500
        self.opt = tf.keras.optimizers.RMSprop(lr=0.001)

        self.loss = tf.keras.losses.categorical_crossentropy
        self.model = self.getBinaryClassifier()

        self.OC = self.getOC()

    def getBinaryClassifier(self):
        model = Sequential(name="binary class")
        model.add(Dense(128, activation='elu', input_dim=self.input_dim))
        model.add(Dropout(.05))
        model.add(Dense(2, activation='softmax'))
        return model

    def getOC(self):
        inputs = tf.keras.Input((self.input_dim,))
        out = tf.keras.layers.Dense(self.k, activation='relu', use_bias=False, name='certificates')(inputs)
        model = tf.keras.models.Model(inputs=[inputs], outputs=out, name='OC')
        return model

    def gradOC(self, inputs):
        with tf.GradientTape() as tape:
            y_hat = self.OC(inputs, training=True)

            # compute the loss
            error = tf.math.reduce_mean(tf.math.square(y_hat))
            error = tf.cast(error, dtype=tf.dtypes.float64)

            W = self.OC.layers[1].get_weights()[0]  # Equation 4.
            W = tf.linalg.matmul(tf.transpose(W), W)
            W = tf.cast(W, dtype=tf.dtypes.float64)

            penalty = tf.math.square(W - tf.eye(self.k, dtype=tf.dtypes.float64)) * 10
            penalty = tf.math.reduce_mean(penalty)

            error = error + penalty

        return error, tape.gradient(error, self.OC.trainable_variables)

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

    def trainOCBatch(self, inputs) -> float:
        """ Calculates loss and gradients for batch of data and applies update to weights """
        loss_value, grads = self.gradOC(inputs)
        self.opt.apply_gradients(zip(grads, self.OC.trainable_variables))
        return loss_value

    def trainOC(self, dataset, batch_size):
        remainder_samples = dataset.shape[0] % batch_size  # Calculate number of remainder samples from batches
        total_loss = []

        # Run batches
        print("Training OC")
        for i in tqdm(range(300)):
            for batch in range(floor(dataset.shape[0] / batch_size)):
                X = dataset[batch_size * batch:batch_size * (batch + 1), :]
                loss = self.trainOCBatch(X)
                total_loss.append(loss)

            # Run remainders
            if remainder_samples > 0:
                X = dataset[(-1) * remainder_samples:, :]
                loss = self.trainOCBatch(X)
                total_loss.append(loss)

        val_avg_loss = sum(total_loss) / len(total_loss)
        print("OC loss: {}".format(val_avg_loss))

    def trainBinaryClassifier(self, dataset, batch_size):
        remainder_samples = dataset.shape[0] % batch_size  # Calculate number of remainder samples from batches
        total_loss = []

        # Run batches
        print("Training DAL Binary Classifier")
        for i in tqdm(range(50)):
            for batch in range(floor(dataset.shape[0] / batch_size)):
                data = dataset[batch_size * batch:batch_size * (batch + 1), :]
                X, y = data[:, :-2], data[:, -2:]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            # Run remainders
            if remainder_samples > 0:
                data = dataset[(-1) * remainder_samples:, :]
                X, y = data[:, :-2], data[:, -2:]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            total_loss = list(chain(*total_loss))
            val_avg_loss = sum(total_loss) / len(total_loss)
            total_loss = []
        print("DAL binary classifier loss: {}".format(val_avg_loss))

    def inferBinaryClassifier(self, inputs):
        yh = self.model(inputs)
        return yh

    def inferOC(self, inputs):
        yh = self.OC(inputs)
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

        # Calculate OC(x) values
        bc_vals = yh.iloc[:, 1].values
        oc_vals = yh.iloc[:, -1].values

        yh_col_names = ["bc", "oc", "ID"]
        yh = pd.concat([pd.DataFrame(bc_vals), pd.DataFrame(oc_vals), pd.DataFrame(cache)], axis=1)
        yh.columns = yh_col_names

        # Get ids
        yh = yh.sort_values(by=['bc'])
        median_index = yh[yh["bc"] == yh["bc"].quantile(.95, interpolation='lower')]
        median_index = median_index.index.values[0]
        yh = yh.iloc[median_index:, :]

        yh = yh.sort_values(by=['oc'])
        n_largest = yh.nlargest(n, 'oc')
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


class clusterDAL(alAlgo):
    """
    DAL(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.

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

    def __init__(self, input_dim=None, n_cluster=4):
        super().__init__(algo_name="clusterDAL")
        self.predict_to_sample = False
        self.feature_set = True
        self.single_output = False
        self.k = n_cluster

        if input_dim == None:
            raise ValueError("Must pass input dim as int to use DAL")
        self.input_dim = input_dim
        self.model = self.getBinaryClassifier()

        self.opt = tf.keras.optimizers.RMSprop(lr=0.001)
        self.loss = tf.keras.losses.categorical_crossentropy

    def getBinaryClassifier(self):
        model = Sequential(name="BC")
        model.add(Dense(256, activation='elu', input_dim=self.input_dim))
        model.add(Dropout(.1))
        model.add(LayerNormalization(axis=1))
        model.add(Dense(2, activation='softmax'))

        return model

    @tf.function
    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(self.model(inputs, training=True), targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    @tf.function
    def trainBatch(self, inputs, targets) -> float:
        """ Calculates loss and gradients for batch of data and applies update to weights """
        loss_value, grads = self.grad(inputs, targets)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    @tf.function
    def predict(self, inputs):
        """ Used for predicting with model but does not have labels """
        yh = self.model(inputs)
        return yh

    def trainBinaryClassifier(self, dataset, batch_size):
        remainder_samples = dataset.shape[0] % batch_size  # Calculate number of remainder samples from batches
        total_loss = []

        # Run batches
        print("Training DAL Binary Classifier")
        for i in tqdm(range(100)):
            for batch in range(floor(dataset.shape[0] / batch_size)):
                data = dataset[batch_size * batch:batch_size * (batch + 1), :]
                X, y = data[:, :-2], data[:, -2:]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            # Run remainders
            if remainder_samples > 0:
                data = dataset[(-1) * remainder_samples:, :]
                X, y = data[:, :-2], data[:, -2:]
                loss = self.trainBatch(X, y)
                total_loss.append(loss)

            total_loss = list(chain(*total_loss))
            val_avg_loss = sum(total_loss) / len(total_loss)
            total_loss = []
        print("DAL binary classifier loss: {}".format(val_avg_loss))

    def inferBinaryClassifier(self, inputs):
        yh = self.model(inputs)
        return yh

    def resetBinayClassifier(self):
        pass

    def cluster(self, data, cache):
        k_means = KMeans(init='k-means++', n_clusters=self.k, n_init=10)
        k_means.fit(data)
        labels = k_means.labels_

        groups = []
        for i in range(self.k):
            samples = np.where(labels == i)
            samples = samples[0]
            samples = samples.astype(int)
            samples_id = [cache[i] for i in samples]
            groups.append(samples_id)

        return groups

    def __call__(self, cache: list, n: int, yh, clusters: list) -> list:

        # Check if embedded cache, then cache is available for the round
        if any(isinstance(i, list) for i in cache):
            try:
                cache = cache[self.round]
            except:
                raise ValueError("Active Learning Algo has iterated through each round\'s unlabled cache.")

        # Check if sample size is to large for cache
        if len(cache) < n:
            raise ValueError("Sample size n is larger than length of round's cache")

        # How many samples come from each group
        cuts = floor(n / self.k)
        cuts = [cuts for i in range(self.k)]
        cuts[-1] = cuts[-1] + n % self.k

        # Calculate values
        yh_vals = yh.iloc[:, 1].values
        yh_col_names = ["yh", "ID"]
        yh = pd.concat([pd.DataFrame(yh_vals), pd.DataFrame(cache)], axis=1)
        yh.columns = yh_col_names

        # Get largest values from each batch
        batch = []
        for i in range(self.k):
            temp_df = yh[yh['ID'].isin(clusters[i])]
            n_largest = temp_df.nlargest(n, 'yh')
            mini_batch = n_largest["ID"].to_list()
            mini_batch = mini_batch[:cuts[i]]
            batch.extend(mini_batch)

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("\n")
        print("Round {} selected samples: {}".format(self.round, batch))
        print("\n")

        # Increment round
        self.round += 1

        return batch


#######################################################


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding of input data."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class modelVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, n_pixels, Beta=1, **kwargs):
        super(modelVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.n_pixels = n_pixels
        self.Beta = Beta

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= self.n_pixels
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss * self.Beta
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(data, reconstruction)
        )
        reconstruction_loss *= self.n_pixels
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss * self.Beta
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction


class VAE(alAlgo):
    """
    VAE(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.

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

    def __init__(self, input_dim=None, codings_size=10, Beta=1):
        super().__init__(algo_name="VAE")
        self.predict_to_sample = False
        self.feature_set = True
        self.single_output = False

        if input_dim == None:
            raise ValueError("Must pass input dim as int to use DAL")
        self.input_dim = input_dim
        self.latent_dim = codings_size

        self.opt = tf.keras.optimizers.RMSprop(lr=0.001)
        self.Beta = Beta

    def getVAE(self):
        encoder_inputs = tf.keras.Input((self.input_dim,))
        x = Dense(80, activation="selu")(encoder_inputs)
        x = Dense(40, activation="selu")(x)
        x = Dense(20, activation="selu")(x)
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = Input(shape=[self.latent_dim])
        x = Dense(20, activation="selu")(latent_inputs)
        x = Dense(40, activation="selu")(x)
        x = Dense(80, activation="selu")(x)
        decoder_outputs = Dense(self.input_dim, )(x)
        decoder = tf.keras.models.Model(inputs=[latent_inputs], outputs=[decoder_outputs])

        vae = modelVAE(encoder, decoder, self.input_dim, Beta=self.Beta)
        vae.compile(optimizer=self.opt)

        return vae

    def trainBatch(self, inputs) -> float:
        """ Calculates loss and gradients for batch of data and applies update to weights """
        loss_value = self.model.train_on_batch(inputs)
        print(loss_value)
        return loss_value

    def predict(self, inputs):
        """ Used for predicting with model but does not have labels """
        yh = self.model(inputs)
        return yh

    def trainVAE(self, dataset, batch_size):
        self.model = self.getVAE()
        self.model.fit(dataset, epochs=15, batch_size=batch_size)

    def inferBinaryClassifier(self, inputs):
        yh = self.model(inputs)
        yh = np.abs(yh - inputs)
        yh = np.sum(yh, axis=-1)
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

        # Calculate values
        yh_vals = yh.values

        yh_col_names = ["yh", "ID"]
        yh = pd.concat([pd.DataFrame(yh_vals), pd.DataFrame(cache)], axis=1)
        yh.columns = yh_col_names

        # Get ids
        yh = yh.sort_values(by=['yh'])
        yh_vals = yh.iloc[:, 0].values
        yh_dist = yh_vals / np.sum(yh_vals)
        cache = yh.iloc[:, 1].values
        # n_largest = yh.nlargest(n, 'yh')
        # batch = n_largest["ID"].to_list()
        batch = np.random.choice(cache, n, p=yh_dist, replace=False)

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("\n")
        print("Round {} selected samples: {}".format(self.round, batch))
        print("\n")

        # Increment round
        self.round += 1

        return batch


#######################################################


class MemAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(MemAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, x):
        with tf.GradientTape() as tape:
            z_hat, w_hat = self.encoder(x)
            x_hat = self.decoder(z_hat)

            mse = tf.reduce_sum(tf.square(x - x_hat))
            mem_etrp = tf.reduce_sum((-w_hat) * tf.math.log(w_hat + 1e-12))

            loss = tf.reduce_mean(mse + (0.0002 * mem_etrp))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss}

    def test_step(self, data):
        x, y = data[0], data[1]
        z_hat, w_hat = self.encoder(x)
        x_hat = self.decoder(z_hat)

        mse = tf.reduce_sum(tf.square(x - x_hat))
        mem_etrp = tf.reduce_sum((-w_hat) * tf.math.log(w_hat + 1e-12))
        loss = tf.reduce_mean(mse + (0.0002 * mem_etrp))

        return {"loss": loss}

    def call(self, inputs):
        z_hat, w_hat = self.encoder(inputs)
        x_hat = self.decoder(z_hat)
        return x_hat, z_hat, w_hat


class Memory(layers.Layer):
    def __init__(self, mem_dim, fea_dim):
        super(Memory, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(self.mem_dim, self.fea_dim), dtype="float32"),
            trainable=True,
        )

    def cosine_sim(self, x1, x2):
        num = tf.linalg.matmul(x1, tf.transpose(x2, perm=[1, 0]))
        denom = tf.linalg.matmul(x1 ** 2, tf.transpose(x2, perm=[1, 0]) ** 2)
        w = (num + 1e-12) / (denom + 1e-12)
        return w

    def call(self, z):
        cosim = self.cosine_sim(x1=z, x2=self.w)
        atteniton = tf.nn.softmax(cosim)

        lam = 1 / self.mem_dim  # deactivate the 1/N of N memories.

        addr_num = tf.keras.activations.relu(atteniton - lam) * atteniton
        addr_denum = tf.abs(atteniton - lam) + 1e-12
        memory_addr = addr_num / addr_denum
        renorm = tf.clip_by_value(memory_addr, 1e-12, 1 - (1e-12))
        z_hat = tf.linalg.matmul(renorm, self.w)

        return z_hat, renorm


class MemAE_AL(alAlgo):
    """
    MemAE(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.

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

    def __init__(self, input_dim=None, codings_size=10):
        super().__init__(algo_name="MemAE AL")
        self.predict_to_sample = False
        self.feature_set = True
        self.single_output = False

        if input_dim == None:
            raise ValueError("Must pass input dim as int to use DAL")
        self.input_dim = input_dim
        self.latent_dim = codings_size

        self.opt = tf.keras.optimizers.RMSprop(lr=0.001)

    def getMemAE(self):
        encoder_inputs = tf.keras.Input((self.input_dim,))
        x = Dense(80, activation="selu")(encoder_inputs)
        x = Dense(40, activation="selu")(x)
        x = Dense(self.latent_dim, activation="selu")(x)
        z_hat, att = Memory(300, self.latent_dim)(x)
        encoder = tf.keras.Model(encoder_inputs, [z_hat, att], name="encoder")

        latent_inputs = tf.keras.Input(shape=(self.latent_dim))
        x = layers.Dense(self.latent_dim, activation="selu")(latent_inputs)
        x = layers.Dense(40, activation="selu")(x)
        x = layers.Dense(80, activation="selu")(x)
        decoder_outputs = layers.Dense(120, activation="selu")(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

        model = MemAE(encoder, decoder)
        model.compile(optimizer=tf.keras.optimizers.Adam())

        return model

    def trainBatch(self, inputs) -> float:
        """ Calculates loss and gradients for batch of data and applies update to weights """
        loss_value = self.model.train_on_batch(inputs)
        print(loss_value)
        return loss_value

    def predict(self, inputs):
        """ Used for predicting with model but does not have labels """
        yh = self.model(inputs)
        return yh

    def trainVAE(self, dataset, batch_size):
        self.model = self.getMemAE()
        self.model.fit(dataset, epochs=25, batch_size=batch_size)

    def inferBinaryClassifier(self, inputs):
        yh, _, _ = self.model(inputs)
        # yh = np.abs(yh - inputs)
        # yh = np.sum(yh, axis=-1)
        # Paper says to use mse as ranking for outliers
        yh = np.square(yh - inputs)
        yh = np.mean(yh, axis=-1)
        return yh

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

        # Calculate values
        yh_vals = yh.values

        yh_col_names = ["yh", "ID"]
        yh = pd.concat([pd.DataFrame(yh_vals), pd.DataFrame(cache)], axis=1)
        yh.columns = yh_col_names

        # Get ids
        yh = yh.sort_values(by=['yh'])
        # yh_vals = yh.iloc[:,0].values
        # yh_dist = yh_vals/np.sum(yh_vals)
        # cache = yh.iloc[:,1].values
        n_largest = yh.nlargest(n, 'yh')
        batch = n_largest["ID"].to_list()
        # batch = np.random.choice(cache, n, p=yh_dist,replace=False)

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("\n")
        print("Round {} selected samples: {}".format(self.round, batch))
        print("\n")

        # Increment round
        self.round += 1

        return batch


class MemAE_Binary_AL(alAlgo):
    """
    MemAE(alAlgo) Documentation:
    --------------------------

    Purpose
    ----------
    Custom active learning class, inherits alAlgo class.

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

    def __init__(self, input_dim=None, codings_size=10):
        super().__init__(algo_name="MemAE_Binary")
        self.predict_to_sample = False
        self.feature_set = True
        self.single_output = False

        if input_dim == None:
            raise ValueError("Must pass input dim as int to use DAL")
        self.input_dim = input_dim
        self.latent_dim = codings_size

        self.opt = tf.keras.optimizers.RMSprop(lr=0.001)

    def getMemAE(self):
        encoder_inputs = tf.keras.Input((self.input_dim,))
        x = Dense(80, activation="selu")(encoder_inputs)
        x = Dense(40, activation="selu")(x)
        x = Dense(self.latent_dim, activation="selu")(x)
        z_hat, att = Memory(500, self.latent_dim)(x)
        encoder = tf.keras.Model(encoder_inputs, [z_hat, att], name="encoder")

        latent_inputs = tf.keras.Input(shape=(self.latent_dim))
        x = layers.Dense(30, activation="selu")(latent_inputs)
        x = layers.Dense(40, activation="selu")(x)
        x = layers.Dense(80, activation="selu")(x)
        decoder_outputs = layers.Dense(self.input_dim, activation="selu")(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

        model = MemAE(encoder, decoder)
        model.compile(optimizer=tf.keras.optimizers.Adam())

        return model

    def getBinaryClassifier(self):
        model = Sequential(name="BC")
        model.add(Dense(15, activation='elu', input_dim=self.latent_dim))
        model.add(Dropout(.1))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=self.opt, loss=tf.keras.losses.BinaryCrossentropy())
        return model

    def trainBatch(self, inputs) -> float:
        """ Calculates loss and gradients for batch of data and applies update to weights """
        loss_value = self.model.train_on_batch(inputs)
        return loss_value

    def predict(self, inputs):
        """ Used for predicting with model but does not have labels """
        yh = self.model(inputs)
        return yh

    def trainVAE(self, dataset, batch_size):
        self.model = self.getMemAE()
        self.model.fit(dataset, epochs=50, batch_size=batch_size)

    def trainBC(self, dataset, batch_size):
        self.modelBC = self.getBinaryClassifier()
        X, y = dataset[:, :-2], dataset[:, -2:]
        self.modelBC.fit(X, y, epochs=20, batch_size=batch_size)

    def inferBinaryClassifier(self, inputs):
        yh = self.modelBC(inputs)
        return yh

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

        # Get ids
        yh = yh.sort_values(by=['yh'])
        yh_vals = yh.iloc[:, 0].values
        yh_dist = yh_vals / np.sum(yh_vals)
        cache = yh.iloc[:, 1].values
        # n_largest = yh.nlargest(n, 'yh')
        # batch = n_largest["ID"].to_list()
        batch = np.random.choice(cache, n, p=yh_dist, replace=False)

        plt.plot(range(0,len(h_dist)),h_dist)
        input("p")

        # Log which samples were used for that round
        self.sample_log[str(self.round)] = batch

        print("\n")
        print("Round {} selected samples: {}".format(self.round, batch))
        print("\n")

        # Increment round
        self.round += 1

        return batch


#######################################################


class SpectralNormalization(tf.keras.layers.Wrapper):
    """Performs spectral normalization on weights.
    This wrapper controls the Lipschitz constant of the layer by
    constraining its spectral norm, which can stabilize the training of GANs.
    See [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).
    ```python
    net = SpectralNormalization(
        tf.keras.layers.Conv2D(2, 2, activation="relu"),
        input_shape=(32, 32, 3))(x)
    net = SpectralNormalization(
        tf.keras.layers.Conv2D(16, 5, activation="relu"))(net)
    net = SpectralNormalization(
        tf.keras.layers.Dense(120, activation="relu"))(net)
    net = SpectralNormalization(
        tf.keras.layers.Dense(n_classes))(net)
    ```
    Arguments:
      layer: A `tf.keras.layers.Layer` instance that
        has either `kernel` or `embeddings` attribute.
      power_iterations: `int`, the number of iterations during normalization.
    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not has `kernel` or `embeddings` attribute.
    """

    @typechecked
    def __init__(self, layer: tf.keras.layers, power_iterations: int = 1, **kwargs):
        super().__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations
        self._initialized = False

    def build(self, input_shape):
        """Build `Layer`"""
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor "
                "'embeddings'".format(type(self.layer).__name__)
            )

        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="sn_u",
            dtype=self.w.dtype,
        )

    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            self.normalize_weights()

        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    @tf.function
    def normalize_weights(self):
        """Generate spectral normalized weights.
        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
                u = tf.math.l2_normalize(tf.matmul(v, w))

            sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

            self.w.assign(self.w / sigma)
            self.u.assign(u)

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}
