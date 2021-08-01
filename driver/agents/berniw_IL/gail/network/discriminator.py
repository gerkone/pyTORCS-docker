import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

from torcs_client.utils import SimpleLogger as log

class Discriminator:
    """
    discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a)
    """
    def __init__(self, state_dims, action_dims, batch_size, lr = 5e-6, load_dir = ""):
        self.batch_size = batch_size
        self.lr = lr

        self.state_dims = state_dims
        self.action_dims = action_dims

        try:
            # load model if present
            self.expert_model = tf.keras.models.load_model(load_dir + "/expert_discriminator")
            self.policy_model = tf.keras.models.load_model(load_dir + "/policy_discriminator")
            log.info("Loaded saved discriminator models")
        except:
            self.expert_model = self.build_network()
            self.expert_model.summary()
            self.policy_model = self.build_network()
            self.policy_model.summary()

            self.policy_model.set_weights(self.expert_model.get_weights())

        self.optimizer = Adam(self.lr)

    def build_network(self):
        # -- state input --
        state_input_layer = Input(shape=(self.state_dims))
        # -- action input --
        action_input_layer = Input(shape=(self.action_dims))

        concat = Concatenate()([state_input_layer, action_input_layer])

        fcl = Dense(32, activation = "relu")(concat)
        fcl = Dense(32, activation = "relu")(fcl)
        fcl = Dense(32, activation = "relu")(fcl)

        # -- output layer --
        output_layer = Dense(1, activation = "sigmoid")(fcl)
        model = Model([state_input_layer, action_input_layer], output_layer)

        return model

    @tf.function
    def train(self, expert_s, expert_a, policy_s, policy_a):
        with tf.GradientTape(persistent = True) as tape:
            P_expert = self.expert_model([expert_s, expert_a])
            P_policy = self.policy_model([policy_s, policy_a])
            loss_expert = tf.reduce_mean(tf.math.log(tf.clip_by_value(P_expert, 1e-10, 1)))
            loss_policy = tf.reduce_mean(tf.math.log(tf.clip_by_value(1 - P_policy, 1e-10, 1)))
            loss = loss_expert + loss_policy
            loss = -loss

        gradient_expert = tape.gradient(loss, self.expert_model.trainable_variables)
        gradient_policy = tape.gradient(loss, self.policy_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_expert, self.expert_model.trainable_variables))
        self.optimizer.apply_gradients(zip(gradient_policy, self.policy_model.trainable_variables))
        return loss

    @tf.function
    def get_reward(self, s, a):
        """
        log(P(expert|s,a))
        """
        with tf.GradientTape() as tape:
            P_policy = self.policy_model([s, a])
            reward = tf.math.log(tf.clip_by_value(P_policy, 1e-10, 1))

        return reward
