import tensorflow as tf
import os
from importlib import import_module

class zooKeeper():
    def __init__(self, dataName, modelName, loss=None, optimizer=None, metrics=None):
        self.dataName = dataName
        self.modelName = modelName

        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        self.modelObject = None

        # TODO: make try to see if dataName is present and check for model by name of modelName


        self.getModel()

    def getModel(self):
        try:
            print("-" * 20)
            print("Found model {}".format(self.modelName))
            print("-" * 20)

            modelClass = getattr(import_module("Zoo.Models."+self.modelName), self.modelName)
            self.modelObject = modelClass()


        except:
            print("Could not properly load the model class for {}.".format(self.modelName))
            raise




"""

# Example class for models in Zoo/Models

class tfModel():
    def __init__(self):
        self.dummy = None
     
    def loadModel(self):
        pass 
    
    def trainBatch(self):
        pass
    
    def predict(self):
        pass 
    
    def saveModel(self):
        pass 
    
    def loadModel(self):
        pass 

"""
