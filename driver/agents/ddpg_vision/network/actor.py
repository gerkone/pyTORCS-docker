import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Add, BatchNormalization, Activation, Multiply
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

"""
Actor network:
stochastic funcion approssimator for the deterministic policy map u : S -> A
(with S set of states, A set of actions)
"""
class Actor(object):
    def __init__(self, state_dims, action_dims, lr, batch_size, tau, fcl1_size,
            fcl2_size, upper_bound, stack_depth, img_height, img_width, encoder):
        self.state_dims = state_dims
        self.action_dims = action_dims
        # learning rate
        self.lr = lr
        self.batch_size = batch_size
        # polyak averaging speed
        self.tau = tau
        self.fcl1_size = fcl1_size
        self.fcl2_size = fcl2_size
        self.upper_bound = upper_bound

        self.stack_depth = stack_depth
        self.img_height = img_height
        self.img_width = img_width

        self.model = self.build_network(encoder.model)
        # self.model.summary()
        #duplicate model for target
        self.target_model = self.build_network(encoder.model)
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = Adam(self.lr)



    def build_network(self, encoder_model):
        """
        Builds the model. Consists of two fully connected layers with batch norm.
        """
        # -- input layer --
        state_input_layer = Input(shape = self.state_dims, name = "State_in")
        # -- first fully connected layer --
        fcl1 = Dense(self.fcl1_size, name = "First_FCL")(state_input_layer)
        fcl1 = Activation("relu", name = "ReLU_1")(fcl1)
        # -- second fully connected layer --
        fcl2 = Dense(self.fcl2_size, name = "Second_FCL")(fcl1)
        fcl2 = Activation("relu", name = "ReLU_2")(fcl2)
        # -- embedded encoder as submodel --
        encoder_input_layer = Input(shape = (self.stack_depth, self.img_height, self.img_width, 3))
        encoder = encoder_model(encoder_input_layer)

        fcl_encoder = Dense(self.fcl2_size, name = "Encoder_FCL")(encoder)
        fcl_encoder = Activation("relu")(fcl_encoder)

        embed = Add()([fcl_encoder, fcl2])
        embed = Activation("relu")(embed)
        # -- output layer --
        f3 = 0.0001
        output_layer = Dense(*self.action_dims, kernel_initializer=RandomUniform(-f3, f3),
            bias_initializer=RandomUniform(-f3, f3), kernel_regularizer=tf.keras.regularizers.l2(0.01))(embed)
        #scale the output (after the activation)
        output_layer = Multiply()([output_layer, tf.ones_like(output_layer) * self.upper_bound])

        model = Model([state_input_layer, encoder_input_layer], output_layer, name = "Actor")
        return model

    @tf.function
    def train(self, states, frames, critic_model):
        """
        Update the weights with the new critic evaluation
        """
        with tf.GradientTape() as tape:
            actions = self.model([states, frames], training=True)
            q_value = critic_model([states, frames, actions], training=True)
            # loss = - (minus) mean critic value
            loss = -tf.math.reduce_mean(q_value)
        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return loss

    def update_target(self):
        """
        Update the target weights using tau as speed. The tracking function is
        defined as:
        target = tau * weights + (1 - tau) * target
        """
        # faster updates woth graph mode
        self._transfer(self.model.variables, self.target_model.variables)

    @tf.function
    def _transfer(self, model_weights, target_weights):
        for (weight, target) in zip(model_weights, target_weights):
            #update the target values
            target.assign(weight * self.tau + target * (1 - self.tau))
