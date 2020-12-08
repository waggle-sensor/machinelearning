import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from Zoo.zoo import customModel


#######################################################

class birdsCNN(customModel):
    """
    birdsCNN() Documentation
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
    Purpose: Load keras model configured for cifar10 data set. If you desire to modify the
            architecture, do so here.

    def trainBatch(self, inputs, targets):
    Purpose: Update model's weights by passing through inputs to the models and calculating loss
            against target values. trainBatch() calls another method in mnistCNN() named grad().
            grad() is what calculates the loss and gradient yet, trainBatch() applies the gradient
            to the model to update the weights. Refer to the following link for a general demo
            of how tensorflow allows for custom training: https://www.tensorflow.org/tutorials/customization/
            custom_training_walkthrough .

    """
    __slots__ = ('loss', 'opt', 'metrics', 'input_shape', 'num_classes', 'model','datasett')

    def __init__(self, loss=None, optimizer=None, metrics=None,dataset=None):
        self.input_shape = (200, 200, 3)  # tf.keras.layers.Conv2D input shape (batch_size, height, width, channels)
        self.num_classes = 200
        self.dataset = "Birds"

        # Loss function for the model, takes loss functions from tf.keras.losses
        if loss == None:
            #self.loss = tf.keras.losses.categorical_crossentropy
            self.loss = tf.keras.losses.cosine_similarity
        else:
            self.loss = loss

        # Optimizer for the model, takes optimizers from tf.keras.optimizers
        if optimizer == None:
            self.opt = tf.keras.optimizers.Adam(clipvalue=0.5)
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

        #model = Sequential(name="birdsCNN")
        #model.add(SeparableConv2D(16, kernel_size=(4, 4), activation='relu',input_shape=self.input_shape))
        #model.add(MaxPooling2D(pool_size=(3, 3)))
        #model.add(LayerNormalization())
        #model.add(Dropout(.2))
        #model.add(SeparableConv2D(32, (4, 4), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(LayerNormalization())
        #model.add(Dropout(.2))
        #model.add(SeparableConv2D(64, kernel_size=(2, 2), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(LayerNormalization())
        #model.add(Dropout(.2))
        #model.add(SeparableConv2D(128, (2, 2), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(LayerNormalization())
        #model.add(Dropout(.2))
        #model.add(Flatten())
        #model.add(Dense(400, activation='relu'))
        #model.add(Dense(self.num_classes, activation='softmax'))

        base_model = tf.keras.applications.Xception(
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            input_shape=(200, 200, 3),
            include_top=False,
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(200, 200, 3))
        x = base_model(inputs,training=False)
        x = tf.keras.layers.SeparableConv2D(128, (1, 1), activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
        outputs = tf.keras.layers.Dense(200,activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        print("Successfully built the model")

        return model
