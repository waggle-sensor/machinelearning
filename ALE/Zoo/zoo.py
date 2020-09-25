# Import modules
from importlib import import_module
import abc
import tensorflow as tf


#######################################################

class zooKeeper():
    def __init__(self, modelName, show_model=False, loss=None, optimizer=None, metrics=None):
        self.modelName = modelName

        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        self.modelObject = None

        self.getModel(show_model)

    def getModel(self, show_model):
        try:
            print("-" * 20)
            print("Found model {}".format(self.modelName))
            print("-" * 20)

            # TODO: make logic for if it is a pre-built class or not
            #modelClass = getattr(import_module("Zoo.Models." + self.modelName), self.modelName)
            modelClass = getattr(import_module("Zoo." + self.modelName), self.modelName)
            self.modelObject = modelClass(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

            # Display model layers, shapes, and number of parameters
            if show_model == True:
                print("\n")
                print("-" * 20)
                print(self.modelObject.model.summary())
                print("-" * 20)

        except:
            print("Could not properly load the model class for {}.".format(self.modelName))
            raise


#######################################################

class customModel(metaclass=abc.ABCMeta):
    def __init__(self):
        self.model = None
        self.loss = None
        self.opt = None
        self.metrics = None

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'loadModel') and
                callable(subclass.loadModel) or
                NotImplemented)

    def testLoad(self):
        print("-" * 20)
        print("Model class successfully loaded!")
        print("-" * 20)

    @abc.abstractmethod
    def loadModel(self) -> tf.keras.Sequential():
        raise NotImplementedError

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

    def getMetrics(self, yh, targets):
        scores = {}
        for i, metric in enumerate(self.metrics):
            metric.reset_states()
            metric.update_state(yh, targets)
            scores[metric.name] = metric.result().numpy()
            metric.reset_states()
        return scores

    def eval(self, inputs, targets):
        """ Used for predicting with model and checking error when labels are provided """
        yh = self.predict(inputs)
        loss_value = self.loss(yh, targets)
        metric_scores = self.getMetrics(yh, targets)
        return loss_value, metric_scores, yh
