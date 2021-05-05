import numpy as np
import keras.backend as K
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Add, Flatten
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

from torcs_client.utils import SimpleLogger as log

class Critic(object):
    """
    Critic network:
    stochastic funcion approssimator for the Q value function C : SxA -> R
    (with S set of states, A set of actions)
    """
    def __init__(self, state_dims, action_dims, lr, batch_size, tau,
                fcl1_size, fcl2_size, save_dir):
        self.state_dims = state_dims
        self.action_dims = action_dims
        # learning rate
        self.lr = lr
        self.batch_size = batch_size
        # polyak averaging speed
        self.tau = tau
        self.fcl1_size = fcl1_size
        self.fcl2_size = fcl2_size

        try:
            # load model if present
            self.model = tf.keras.models.load_model(save_dir + "/critic")
            self.target_model = tf.keras.models.load_model(save_dir +"/critic_target")
            log.info("Loaded saved critic models")
        except:
            self.model = self.build_network()
            #duplicate model for target
            self.target_model = self.build_network()
            self.target_model.set_weights(self.model.get_weights())
            self.model.summary()
        self.optimizer = Adam(self.lr)

    def build_network(self):
        """
        Builds the model,
        non-sequential, state and action as inputs:
        two state fully connected layers and one action fully connected layer.
        Action introduced after the second state layer, as specified in the paper
        """
        # -- state input --
        state_input_layer = Input(shape=(self.state_dims))
        # -- action input --
        action_input_layer = Input(shape=(self.action_dims))
        # -- hidden fully connected layers --
        f1 = 1. / np.sqrt(self.fcl1_size)
        fcl1 = Dense(self.fcl1_size, kernel_initializer = RandomUniform(-f1, f1),
                    bias_initializer = "zeros")(state_input_layer)
        fcl1 = BatchNormalization()(fcl1)
        #activation applied after batchnorm
        fcl1 = Activation("relu")(fcl1)
        f2 = 1. / np.sqrt(self.fcl2_size)
        fcl2 = Dense(self.fcl2_size, kernel_initializer = RandomUniform(-f2, f2),
                    bias_initializer = "zeros")(fcl1)
        fcl2 = BatchNormalization()(fcl2)
        #activation applied after batchnorm
        # fcl2 = Activation("linear")(fcl2)
        # Introduce action after the second layer
        action_layer =  Dense(self.fcl2_size, kernel_initializer = RandomUniform(-f2, f2),
                    bias_initializer = "zeros")(action_input_layer)
        action_layer = Activation("relu")(action_layer)
        concat = Add()([fcl2, action_layer])
        concat = Activation("relu")(concat)
        # Outputs single value for give state-action
        f3 = 0.003
        output = Dense(1, kernel_initializer=RandomUniform(-f3, f3),
                    bias_initializer=RandomUniform(-f3, f3),
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(concat)

        model = Model([state_input_layer, action_input_layer], output)
        return model

    @tf.function
    def train(self, states, actions, rewards, terminals, states_n, actor_target, gamma):
        """
        Update the weights with the Q targets. Graphed function for more
        efficient Tensor operations
        """
        with tf.GradientTape() as tape:
            target_actions = actor_target(states_n, training=True)
            q_n = self.target_model([states_n, target_actions], training=True)
            # Bellman equation for the q value
            q_target = rewards + gamma * q_n * (1 - terminals)
            q_value = self.model([states, actions], training=True)
            loss = MSE(q_target, q_value)

        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

    def update_target(self):
        """
        Update the target weights using tau as speed. The tracking function is
        defined as:
        target = tau * weights + (1 - tau) * target
        """
        # faster updates with graph mode
        self._transfer(self.model.variables, self.target_model.variables)

    @tf.function
    def _transfer(self, model_weights, target_weights):
        """
        Target update helper. Applies Polyak averaging on the target weights.
        """
        for (weight, target) in zip(model_weights, target_weights):
            #update the target values
            target.assign(weight * self.tau + target * (1 - self.tau))
