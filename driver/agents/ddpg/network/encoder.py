import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, BatchNormalization, Activation, Multiply
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

class Encoder(object):
    def __init__(self, stack_depth, img_height, img_width):
        self.stack_depth = stack_depth
        self.img_height = img_height
        self.img_width = img_width

        self.model = self.build_network()

    def build_network(self):
        """
        Builds the model. Consists of two fully connected layers with batch norm.
        """
        # -- input layer --
        input_layer = Input(shape = (self.stack_depth, self.img_height, self.img_width, 3))
        conv_1 = Conv2D(16, (3,3), strides=8, padding ="same", activation = "relu", name="conv_1",
            kernel_initializer="glorot_uniform", bias_initializer="zeros")(input_layer)
        conv_2 = Conv2D(32, (3,3), strides=4, padding="same", activation="relu",name="conv_2",
            kernel_initializer="glorot_uniform", bias_initializer="zeros")(conv_1)
        conv_3 = Conv2D(64, (3,3), strides=2, padding="same", activation="relu", name="conv_3",
            kernel_initializer="glorot_uniform", bias_initializer="zeros")(conv_2)
        conv_4 = Conv2D(128, (3,3), strides=1, padding="same", activation="relu", name="conv_4",
            kernel_initializer="glorot_uniform", bias_initializer="zeros")(conv_3)
        output_layer = Flatten()(conv_4)

        model = Model(input_layer, output_layer)
        return model
