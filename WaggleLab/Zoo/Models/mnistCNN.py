import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential

class mnistCNN():
    def __init__(self, loss=None, optimizer=None, metrics=None):
        self.input_shape = (28, 28, 1) # tf.keras.layers.Conv2D input shape (batch_size, height, width, channels)
        self.num_classes = 10

        if loss == None:
            self.loss = tf.keras.losses.categorical_crossentropy
        else:
            self.loss = loss

        if optimizer == None:
            self.opt = tf.keras.optimizers.SGD(lr=0.01)
        else:
            self.opt = optimizer

        if metrics == None:
            self.metrics = ['accuracy']
        else:
            self.metrics = metrics

        self.model = self.loadModel()

    def testLoad(self):
        print("-" * 20)
        print("Model class successfully loaded!")
        print("-"*20)

    def loadModel(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        self.model = model

        print("-" * 20)
        print("Successfully built the model")
        print("-" * 20)

    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(self.model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def trainBatch(self,inputs, targets):
        loss_value, grads = self.grad(inputs, targets)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        pass

    def predict(self,inputs):
        yh = self.model(inputs)
        return yh

    def eval(self,inputs,targets):
        score = self.model.evaluate(inputs,targets)
        return score

    def saveModel(self):
        pass


