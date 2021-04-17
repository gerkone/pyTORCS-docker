import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

from agents.ddpg.utils.replay_buffer import ReplayBuffer
from agents.ddpg.utils.action_noise import OUActionNoise
from agents.ddpg.network.actor import Actor
from agents.ddpg.network.critic import Critic

class DDPG(object):
    """
    DDPG agent
    """
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):

        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        actor_lr = hyperparams["actor_lr"]
        critic_lr = hyperparams["critic_lr"]
        batch_size = hyperparams["batch_size"]
        gamma = hyperparams["gamma"]
        rand_steps = hyperparams["rand_steps"]
        buf_size = int(hyperparams["buf_size"])
        tau = hyperparams["tau"]
        fcl1_size = hyperparams["fcl1_size"]
        fcl2_size = hyperparams["fcl2_size"]

        # action size
        self.n_states = state_dims[0]
        # state size
        self.n_actions = action_dims[0]
        self.batch_size = batch_size

        # environmental action boundaries
        self.lower_bound = action_boundaries[0]
        self.upper_bound = action_boundaries[1]

        # experience replay buffer
        self._memory = ReplayBuffer(buf_size, input_shape = state_dims, output_shape = action_dims)
        # noise generator
        self._noise = OUActionNoise(mu=np.zeros(action_dims))
        # Bellman discount factor
        self.gamma = gamma

        # number of episodes for random action exploration
        self.rand_steps = rand_steps - 1

        # turn off most logging
        logging.getLogger("tensorflow").setLevel(logging.FATAL)

        # date = datetime.now().strftime("%m%d%Y_%H%M%S")
        # path_actor = "./models/actor/actor" + date + ".h5"
        # path_critic = "./models/critic/actor" + date + ".h5"

        # actor class
        self.actor = Actor(state_dims = state_dims, action_dims = action_dims,
                            lr = actor_lr, batch_size = batch_size, tau = tau,
                            upper_bound = self.upper_bound,
                            fcl1_size = fcl1_size, fcl2_size = fcl2_size)
        # critic class
        self.critic = Critic(state_dims = state_dims, action_dims = action_dims,
                            lr = critic_lr, batch_size = batch_size, tau = tau,
                            fcl1_size = fcl1_size, fcl2_size = fcl2_size)

    def get_action(self, state, step):
        """
        Return the best action in the passed state, according to the model
        in training. Noise added for exploration
        """
        #take only random actions for the first episode
        if(step > self.rand_steps):
            noise = self._noise()
            state = np.hstack(list(state.values()))
            state = tf.expand_dims(state, axis = 0)
            action = self.actor.model.predict(state)
            action_p = action + noise
            action_p = action_p[0]
        else:
            #explore the action space quickly
            action_p = np.random.uniform(self.lower_bound, self.upper_bound, self.n_actions)
        #clip the resulting action with the bounds
        action_p = np.clip(action_p, self.lower_bound, self.upper_bound)
        return action_p


    def learn(self, i):
        """
        Fill the buffer up to the batch size, then train both networks with
        experience from the replay buffer.
        """
        actor_loss = -1
        if self._memory.isReady(self.batch_size):
            actor_loss = self.train_helper()
        return actor_loss
    """
    Train helper methods
    train_helper
    train_critic
    train_actor
    get_q_targets  Q values to train the critic
    get_gradients  policy gradients to train the actor
    """

    def train_helper(self):
        # get experience batch
        states, actions, rewards, terminal, states_n = self._memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, dtype=tf.float32)
        states_n = tf.convert_to_tensor(states_n)

        # train the critic before the actor
        self.train_critic(states, actions, rewards, terminal, states_n)
        actor_loss = self.train_actor(states)
        #update the target models
        self.critic.update_target()
        self.actor.update_target()
        return actor_loss

    def train_critic(self, states, actions, rewards, terminal, states_n):
        """
        Use updated Q targets to train the critic network
        """
        # TODO cleaner code, ugly passing of actor target model
        self.critic.train(states, actions, rewards, terminal, states_n, self.actor.target_model, self.gamma)

    def train_actor(self, states):
        """
        Train the actor network with the critic evaluation
        """
        # TODO cleaner code, ugly passing of critic model
        return self.actor.train(states, self.critic.model)

    def remember(self, state, state_new, action, reward, terminal):
        """
        replay buffer interfate to the outsize
        """
        self._memory.remember(state, state_new, action, reward, terminal)
