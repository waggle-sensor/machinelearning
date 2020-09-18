from importlib import import_module


#######################################################

class zooKeeper():
    def __init__(self, dataName, modelName, show_model=False, loss=None, optimizer=None, metrics=None):
        self.dataName = dataName
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
            modelClass = getattr(import_module("Zoo.Models." + self.modelName), self.modelName)
            self.modelObject = modelClass()

            # Display model layers, shapes, and number of parameters
            if show_model == True:
                print("-" * 20)
                print(self.modelObject.model.summary())
                print("-" * 20)

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
