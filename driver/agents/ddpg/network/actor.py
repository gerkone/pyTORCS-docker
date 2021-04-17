import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Lambda
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

"""
Actor network:
stochastic funcion approssimator for the deterministic policy map u : S -> A
(with S set of states, A set of actions)
"""
class Actor(object):
    def __init__(self, state_dims, action_dims, lr, batch_size, tau,
                    fcl1_size, fcl2_size, upper_bound):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.lr = lr
        # learning rate
        self.batch_size = batch_size
        self.tau = tau
        # polyak averaging speed
        self.fcl1_size = fcl1_size
        self.fcl2_size = fcl2_size
        self.upper_bound = upper_bound

        self.model = self.build_network()
        self.model.summary()
        #duplicate model for target
        self.target_model = self.build_network()
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = Adam(self.lr)


    def build_network(self):
        """
        Builds the model. Consists of two fully connected layers with batch norm.
        """
        # -- input layer --
        input_layer = Input(shape = self.state_dims)
        # -- first fully connected layer --
        fcl1 = Dense(self.fcl1_size)(input_layer)
        fcl1 = BatchNormalization()(fcl1)
        #activation applied after batchnorm
        fcl1 = Activation("relu")(fcl1)
        # -- second fully connected layer --
        fcl2 = Dense(self.fcl2_size)(fcl1)
        fcl2 = BatchNormalization()(fcl2)
        #activation applied after batchnorm
        fcl2 = Activation("relu")(fcl2)
        # -- output layer --
        f3 = 0.003
        output_layer = Dense(*self.action_dims, activation="tanh", kernel_initializer = RandomUniform(-f3, f3),
                        bias_initializer = RandomUniform(-f3, f3),
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))(fcl2)
        #scale the output
        output_layer = Lambda(lambda i : i * self.upper_bound)(output_layer)
        # output_layer =  output_layer * self.upper_bound
        model = Model(input_layer, output_layer)
        return model

    @tf.function
    def train(self, states, critic_model):
        """
        Update the weights with the new critic evaluation
        """
        with tf.GradientTape() as tape:
            actions = self.model(states, training=True)
            q_value = critic_model([states, actions], training=True)
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
