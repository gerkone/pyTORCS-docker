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
                fcl1_size, fcl2_size, noise_bound, lower_bound, upper_bound):
        self.state_dims = state_dims
        self.action_dims = action_dims
        # learning rate
        self.lr = lr
        self.batch_size = batch_size
        # polyak averaging speed
        self.tau = tau
        self.fcl1_size = fcl1_size
        self.fcl2_size = fcl2_size

        self.model = self.build_network()
        self.model.summary()
        #duplicate model for target
        self.target_model = self.build_network()
        self.target_model.set_weights(self.model.get_weights())

        self.noise_bound = noise_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        #generate gradient function
        self.optimizer = Adam(self.lr)

    def build_network(self):
        """
        Builds the model,
        non-sequential, state and action as inputs:
        two state fully connected layers and one action fully connected layer.
        Action introduced after the second state layer, as specified in the paper.
        Twin network to approximate two concurring approximators
        """

        # # ---- First Q model -----
        #
        # # -- state input --
        # state_input_layer = Input(shape=(self.state_dims), name='State_in')
        # # -- action input --
        # action_input_layer = Input(shape=(self.action_dims), name='Action_in')
        # # -- hidden fully connected layers --
        # f1 = 1. / np.sqrt(self.fcl1_size)
        # fcl1_q1 = Dense(self.fcl1_size, kernel_initializer = RandomUniform(-f1, f1),
        #             bias_initializer = RandomUniform(-f1, f1), name = "First_FCL_Q1")(state_input_layer)
        # #activation applied after batchnorm
        # fcl1_q1 = Activation("relu", name = "ReLU_1")(fcl1_q1)
        # f2 = 1. / np.sqrt(self.fcl2_size)
        # fcl2_q1 = Dense(self.fcl2_size, kernel_initializer = RandomUniform(-f2, f2),
        #             bias_initializer = RandomUniform(-f2, f2), name = "Second_FCL_Q1")(fcl1_q1)
        # #activation applied after batchnorm
        # # Introduce action after the second layer
        # action_layer_q1 =  Dense(self.fcl2_size, kernel_initializer = RandomUniform(-f2, f2),
        #             bias_initializer = RandomUniform(-f2, f2), name = "Action_FCL_Q1")(action_input_layer)
        # action_layer_q1 = Activation("relu", name = "ReLU_2")(action_layer_q1)
        # concat_q1 = Add(name = "Action_join_layer_Q1")([fcl2_q1, action_layer_q1])
        # concat_q1 = Activation("relu", name = "ReLU_join")(concat_q1)
        # # Outputs single value for give state-action
        # f3 = 0.003
        # output_q1 = Dense(1, kernel_initializer=RandomUniform(-f3, f3),
        #             bias_initializer=RandomUniform(-f3, f3),
        #             kernel_regularizer=tf.keras.regularizers.l2(0.01), name = "Q1")(concat_q1)
        #
        # # ---- Second Q model -----
        #
        # # -- hidden fully connected layers --
        # fcl1_q2 = Dense(self.fcl1_size, kernel_initializer = RandomUniform(-f1, f1),
        #             bias_initializer = RandomUniform(-f1, f1), name = "First_FCL_Q2")(state_input_layer)
        # #activation applied after batchnorm
        # fcl1_q2 = Activation("relu", name = "ReLU_3")(fcl1_q2)
        # fcl2_q2 = Dense(self.fcl2_size, kernel_initializer = RandomUniform(-f2, f2),
        #             bias_initializer = RandomUniform(-f2, f2), name = "Second_FCL_Q2")(fcl1_q2)
        # #activation applied after batchnorm
        # # Introduce action after the second layer
        # action_layer_q2 =  Dense(self.fcl2_size, kernel_initializer = RandomUniform(-f2, f2),
        #             bias_initializer = RandomUniform(-f2, f2), name = "Action_FCL_Q2")(action_input_layer)
        # action_layer_q2 = Activation("relu", name = "ReLU_4")(action_layer_q2)
        # concat_q2 = Add(name = "Action_join_layer_Q2")([fcl2_q2, action_layer_q2])
        # concat_q2 = Activation("relu", name = "ReLU_join_2")(concat_q2)
        # # Outputs single value for give state-action
        # output_q2 = Dense(1, kernel_initializer=RandomUniform(-f3, f3),
        #             bias_initializer=RandomUniform(-f3, f3),
        #             kernel_regularizer=tf.keras.regularizers.l2(0.01), name = "Q2")(concat_q2)
        #
        # output = Concatenate(axis=1)([output_q1, output_q2])
        #
        # model = Model([state_input_layer, action_input_layer], output)
        # return model

        # -- state input --
        state_input_layer = Input(shape=(self.state_dims))
        # -- action input --
        action_input_layer = Input(shape=(self.action_dims))
        # -- hidden fully connected layers --
        f1 = 1. / np.sqrt(self.fcl1_size)
        fcl1 = Dense(self.fcl1_size, kernel_initializer = RandomUniform(-f1, f1),
                    bias_initializer = RandomUniform(-f1, f1))(state_input_layer)
        fcl1 = BatchNormalization()(fcl1)
        #activation applied after batchnorm
        fcl1 = Activation("relu")(fcl1)
        f2 = 1. / np.sqrt(self.fcl2_size)
        fcl2 = Dense(self.fcl2_size, kernel_initializer = RandomUniform(-f2, f2),
                    bias_initializer = RandomUniform(-f2, f2))(fcl1)
        fcl2 = BatchNormalization()(fcl2)
        #activation applied after batchnorm
        # fcl2 = Activation("linear")(fcl2)
        # Introduce action after the second layer
        action_layer =  Dense(self.fcl2_size, kernel_initializer = RandomUniform(-f2, f2),
                    bias_initializer = RandomUniform(-f2, f2))(action_input_layer)
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
            # clipped noise for smooth policy update
            smoothing_noise = tf.random.uniform(actions.shape, -self.noise_bound, self.noise_bound)
            target_actions = tf.clip_by_value(actor_target(states_n, training=True),
                                        self.lower_bound, self.upper_bound)
            # q_n = tf.transpose(self.target_model([states_n, target_actions], training=True))
            # q_1_target = tf.gather(q_n, 0)
            # q_2_target = tf.gather(q_n, 1)

            q_n = self.target_model([states_n, target_actions], training=True)

            # # q target as minimum between the models
            # q_target = tf.math.minimum(q_1_target, q_2_target)

            # Bellman equation for the q value
            q_target = rewards + gamma * q_n * (1 - terminals)

            # q_value = tf.transpose(self.model([states, actions], training=True))
            # q_1 = tf.gather(q_value, 0)
            # q_2 = tf.gather(q_value, 1)
            # Bellman equation for the q value
            q_target = rewards + gamma * q_n * (1 - terminals)
            q_value = self.model([states, actions], training=True)

            # critic loss as sum of losses from both the q function estimators
            # loss = MSE(q_target, q_1) + MSE(q_target, q_2)
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
