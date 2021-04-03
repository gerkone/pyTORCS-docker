import numpy as np
import keras.backend as K
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Add, Concatenate
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

class Critic(object):
    """
    Critic network:
    stochastic funcion approximator for the Q value function C : SxA -> R
    (with S set of states, A set of actions)
    """
    def __init__(self, state_dims, action_dims, lr, batch_size, tau,
                fcl1_size, fcl2_size, noise_bound, lower_bound, upper_bound,
                stack_depth, img_height, img_width, encoder):
        self.state_dims = state_dims
        self.action_dims = action_dims
        # learning rate
        self.lr = lr
        self.batch_size = batch_size
        # polyak averaging speed
        self.tau = tau
        self.fcl1_size = fcl1_size
        self.fcl2_size = fcl2_size

        self.stack_depth = stack_depth
        self.img_height = img_height
        self.img_width = img_width

        self.model = self.build_network(encoder.model)
        self.model.summary()
        #duplicate model for target
        self.target_model = self.build_network(encoder.model)
        self.target_model.set_weights(self.model.get_weights())

        self.noise_bound = noise_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        #generate gradient function
        self.optimizer = Adam(self.lr)

    def build_network(self, encoder_model):
        """
        Builds the model,
        non-sequential, state and action as inputs:
        two state fully connected layers and one action fully connected layer.
        Action introduced after the second state layer, as specified in the paper.
        Twin network to approximate two concurring approximators
        """
        # -- state input --
        state_input_layer = Input(shape=(self.state_dims), name = "State_in")
        # -- embedded encoder as submodel --
        encoder_input_layer = Input(shape = (self.stack_depth, self.img_height, self.img_width, 3), name = "Frame_in")
        encoder = encoder_model(encoder_input_layer)

        fcl_encoder = Dense(self.fcl2_size, name = "Encoder_FCL")(encoder)
        fcl_encoder = Activation("relu")(fcl_encoder)
        # -- action input --
        action_input_layer = Input(shape=(self.action_dims), name = "Action_in")

        # -- hidden fully connected layers --
        f1 = 1. / np.sqrt(self.fcl1_size)
        fcl1 = Dense(self.fcl1_size, kernel_initializer = RandomUniform(-f1, f1),
                bias_initializer = RandomUniform(-f1, f1), name = "First_FCL")(state_input_layer)
        fcl1 = BatchNormalization()(fcl1)
        #activation applied after batchnorm
        fcl1 = Activation("relu")(fcl1)
        f2 = 1. / np.sqrt(self.fcl2_size)
        fcl2 = Dense(self.fcl2_size, kernel_initializer = RandomUniform(-f2, f2),
                bias_initializer = RandomUniform(-f2, f2), name = "Second_FCL")(fcl1)
        fcl2 = BatchNormalization()(fcl2)

        #activation applied after batchnorm
        # fcl2 = Activation("linear")(fcl2)
        # Introduce action after the second layer
        action_layer =  Dense(self.fcl2_size, kernel_initializer = RandomUniform(-f2, f2),
                bias_initializer = RandomUniform(-f2, f2), name = "Action_FCL")(action_input_layer)
        action_layer = Activation("relu")(action_layer)

        embed = Add(name = "Embed")([fcl_encoder, fcl2])
        embed = Activation("relu")(embed)

        concat = Add(name = "Action_concat")([embed, action_layer])
        concat = Activation("relu")(concat)
        # Outputs single value for give state-action
        f3 = 0.003
        output = Dense(1, kernel_initializer=RandomUniform(-f3, f3), bias_initializer=RandomUniform(-f3, f3),
                kernel_regularizer=tf.keras.regularizers.l2(0.01))(concat)

        model = Model([state_input_layer, encoder_input_layer, action_input_layer], output, name = "Critic")
        return model

    @tf.function
    def train(self, states, frames, actions, rewards, terminals, states_n, actor_target, gamma):
        """
        Update the weights with the Q targets. Graphed function for more
        efficient Tensor operations
        """
        with tf.GradientTape() as tape:
            # clipped noise for smooth policy update
            smoothing_noise = tf.random.uniform(actions.shape, -self.noise_bound, self.noise_bound)
            target_actions = tf.clip_by_value(actor_target([states_n, frames], training=True), self.lower_bound, self.upper_bound)
            q_n = self.target_model([states_n, frames, target_actions], training=True)

            q_target = rewards + gamma * q_n * (1 - terminals)
            q_value = self.model([states, frames, actions], training=True)

            # critic loss as sum of losses from both the q function estimators
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
