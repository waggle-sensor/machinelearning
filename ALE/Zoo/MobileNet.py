import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from Zoo.zoo import customModel
import numpy as np


#######################################################

class MobileNet(customModel):
    """
    MobileNet() Documentation
    --------------------------

    Purpose
    ----------
    Custom model that can be called by class zooKeeper from zoo.py

    Attributes
    ----------
    :param loss: loss function for keras model, tf.keras.losses
    :param opt: optimizer for keras model, tf.keras.optimizers
    :param metrics: list of metrics for keras model to track

    Methods
    -------
    def loadModel(self):
    Purpose: Load keras model configured for mnist data set. If you desire to modify the
            architecture, do so here.

    def trainBatch(self, inputs, targets):
    Purpose: Update model's weights by passing through inputs to the models and calculating loss
            against target values. trainBatch() calls another method in mnistCNN() named grad().
            grad() is what calculates the loss and gradient yet, trainBatch() applies the gradient
            to the model to update the weights. Refer to the following link for a general demo
            of how tensorflow allows for custom training: https://www.tensorflow.org/tutorials/customization/
            custom_training_walkthrough .

    """
    __slots__ = ('loss', 'opt', 'metrics', 'input_shape', 'num_classes', 'model', 'dataset')

    def __init__(self, loss=None, optimizer=None, metrics=None, dataset=None):

        self.dataset = dataset
        if dataset == None:
            raise ValueError("dataset not provided. Pass either 'MNIST' or 'CIFAR10")
        elif dataset == "MNIST":
            self.input_shape = (28, 28, 1)
            self.num_classes = 10
        elif dataset == "CIFAR10":
            self.input_shape = (32, 32, 3)
            self.num_classes = 10
        else:
            raise ValueError("dataset is not equal to 'MNIST' or 'CIFAR10' ")

        # Loss function for the model, takes loss functions from tf.keras.losses
        if loss == None:
            self.loss = tf.keras.losses.categorical_crossentropy
        else:
            self.loss = loss

        # Optimizer for the model, takes optimizers from tf.keras.optimizers
        if optimizer == None:
            self.opt = tf.keras.optimizers.Adam(lr=0.001)
        else:
            self.opt = optimizer

        # Metrics for model
        if metrics == None:
            print("didnt get metrics list")
            self.metrics = [tf.keras.metrics.MeanSquaredError()]
        else:
            self.metrics = metrics

        # Declare model
        self.model = self.loadModel()

    def testLoad(self):
        print("-" * 20)
        print("Model class successfully loaded!")
        print("-" * 20)

    def loadModel(self) -> tf.keras.Sequential():
        """ Creates tf.keras model and returns it. Change model architecture here """

        model = Sequential()
        model.add(SeparableConv2D(32, 3, strides=2, input_shape=self.input_shape))
        model.add(Dropout(.2))
        model.add(SeparableConv2D(32, 3))
        model.add(Dropout(.2))
        model.add(SeparableConv2D(64, 2))
        model.add(Dropout(.2))
        model.add(SeparableConv2D(64, 1))
        model.add(Dropout(.2))
        model.add(SeparableConv2D(128, 1))
        model.add(Dropout(.2))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation="softmax"))

        print("Successfully built the model")

        return model
