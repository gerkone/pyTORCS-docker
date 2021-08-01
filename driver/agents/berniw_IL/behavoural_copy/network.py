import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Multiply, BatchNormalization, Activation
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(self, state_dims = 28, action_dims = 2, lr = 1e-3, save_dir = "BC_agent", batch_size = 32, epochs = 6, load = False):
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], False)

        self.batch_size = batch_size
        self.epochs = epochs
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.lr = lr
        # learning rate
        self.batch_size = batch_size

        if load == True:
            # load model if present
            self.model = tf.keras.models.load_model(save_dir)
            print("Loaded saved agent model")
        else:
            self.model = self.build_network()
            self.model.summary()

    def get_action(self, state):
        state = tf.expand_dims(state, axis = 0)
        return self.model.predict(state)[0]

    def build_network(self):
        """
        Builds the model
        """
        # -- input layer --
        input_layer = Input(shape = self.state_dims)
        fcl = Dense(300, activation = "relu")(input_layer)
        fcl = Dense(600, activation = "relu")(fcl)
        fcl = Dense(200, activation = "relu")(fcl)

        # -- output layer --
        output_layer = Dense(self.action_dims, activation = "linear")(fcl)
        model = Model(input_layer, output_layer)

        adam = Adam(learning_rate = self.lr)

        model.compile(loss="mean_squared_error", optimizer = adam)

        return model

    def train(self, input, output):
        self.model.fit(input, output, epochs = self.epochs, batch_size = self.batch_size)
        self.model.save("BC_agent")
