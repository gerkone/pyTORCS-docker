import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Multiply, BatchNormalization, Activation
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

from torcs_client.utils import SimpleLogger as log

class Policy:
    def __init__(self, state_dims, action_dims, batch_size, gamma, lr = 5e-6, load_dir = ""):
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.value_w = 1
        self.entropy_w = 0.01

        try:
            # load model if present
            self.value_network = tf.keras.models.load_model(load_dir + "/value_network")
            self.policy_network = tf.keras.models.load_model(load_dir + "/policy_network")
            # recreate old models
            self.value_network_old = self.build_network(1)
            self.policy_network_old = self.build_network(*self.action_dims)
            self.value_network_old.set_weights(self.value_network.get_weights())
            self.policy_network_old.set_weights(self.policy_network.get_weights())
            log.info("Loaded saved policy and value models")
        except:
            self.value_network = self.build_network(1)
            self.value_network.summary()
            self.policy_network = self.build_network(*self.action_dims)
            self.policy_network.summary()
            # old models, previous
            self.value_network_old = self.build_network(1)
            self.policy_network_old = self.build_network(*self.action_dims)

        self.optimizer = Adam(self.lr)

    def build_network(self, output_dims):
        """
        Builds the model
        """
        # -- input layer --
        input_layer = Input(shape = self.state_dims)
        fcl = Dense(32, activation = "relu")(input_layer)
        fcl = Dense(32, activation = "relu")(fcl)
        fcl = Dense(32, activation = "relu")(fcl)

        # -- output layer --
        if output_dims == 1:
            # value
            output_layer = Dense(output_dims, activation = "linear")(fcl)
        else:
            # policy
            output_layer = Dense(output_dims, activation = "softmax")(fcl)
        model = Model(input_layer, output_layer)

        return model

    def get_action(self, state):
        """
        return the preferred action
        """
        state = tf.expand_dims(state, axis = 0)
        return self.policy_network.predict(state)[0]

    def get_value(self, state):
        """
        return the predicted value
        """
        state = tf.expand_dims(state, axis = 0)
        return self.value_network.predict(state)[0]

    @tf.function
    def train(self, state, action, gae, estimated_rewards, v_preds_next):
        with tf.GradientTape(persistent = True) as tape:

            probs = self.policy_network(state)
            probs = tf.reduce_sum(probs, axis = 1)

            probs_old = self.policy_network_old(state)
            probs_old = tf.reduce_sum(probs_old, axis = 1)

            ratios = tf.math.exp(tf.math.log(tf.clip_by_value(probs, 1e-10, 1.0)) - tf.math.log(tf.clip_by_value(probs_old, 1e-10, 1.0)))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min = 0.8, clip_value_max = 1.2)

            loss_clip = tf.minimum(tf.multiply(gae, ratios), tf.multiply(gae, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)

            # entropy loss
            # entropy = -tf.reduce_sum(probs * tf.math.log(tf.clip_by_value(probs, 1e-10, 1.0)), axis=1)
            # entropy = tf.reduce_mean(entropy, axis=0)

            # value loss
            v_preds = self.value_network(state)
            loss_vf = tf.math.squared_difference(estimated_rewards + self.gamma * v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)

            # clipped loss
            loss = loss_clip - self.value_w * loss_vf # + self.entropy_w  * entropy
            loss = -loss

        gradient_expert = tape.gradient(loss, self.policy_network.trainable_variables)
        gradient_policy = tape.gradient(loss, self.value_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_expert, self.policy_network.trainable_variables))
        self.optimizer.apply_gradients(zip(gradient_policy, self.value_network.trainable_variables))
        return loss
