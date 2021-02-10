import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

from agent.utils.replay_buffer import ReplayBuffer
from agent.utils.action_noise import OUActionNoise
from agent.network.actor import Actor
from agent.network.critic import Critic

class Agent(object):
    """
    DDPG agent
    """
    def __init__(self, state_dims, action_dims, action_boundaries, actor_lr = 1e-5,
                critic_lr = 1e-4, batch_size = 64, gamma = 0.99, actor_update_delay = 2,
                rand_steps = 1, buf_size = 10000, tau = 1e-3, fcl1_size = 400, fcl2_size = 600):
        # action size
        self.n_states = state_dims[0]
        # state size
        self.n_actions = action_dims[0]
        self.batch_size = batch_size
        # experience replay buffer
        self._memory = ReplayBuffer(buf_size, state_dims, action_dims)
        # noise generator
        self._noise = OUActionNoise(mu=np.zeros(action_dims))
        # Bellman discount factor
        self.gamma = gamma
        # environmental action boundaries
        self.lower_bound = action_boundaries[0]
        self.upper_bound = action_boundaries[1]

        # number of episodes for random action exploration
        self.rand_steps = rand_steps - 1

        self.actor_update_delay = actor_update_delay

        # turn off most logging
        logging.getLogger("tensorflow").setLevel(logging.FATAL)

        date = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.path_actor = "./models/actor/actor" + date
        self.path_critic = "./models/critic/actor" + date

        # actor class
        self.actor = Actor(state_dims = state_dims, action_dims = action_dims,
                            lr = actor_lr, batch_size = batch_size, tau = tau,
                            upper_bound = self.upper_bound,
                            fcl1_size = fcl1_size, fcl2_size = fcl2_size)
        # critic class
        self.critic = Critic(state_dims = state_dims, action_dims = action_dims,
                            lr = critic_lr, batch_size = batch_size, tau = tau,
                            lower_bound = self.lower_bound, upper_bound = self.upper_bound,
                            noise_bound = self.upper_bound / 10, fcl1_size = fcl1_size, fcl2_size = fcl2_size)

    def get_action(self, state, step):
        """
        Return the best action in the passed state, according to the model
        in training. Noise added for exploration
        """
        #take only random actions for the first episode
        if(step > self.rand_steps):
            noise = self._noise()
            state = state.reshape(self.n_states, 1).T
            action = self.actor.model.predict(state)[0]
            action_p = action + noise
        else:
            #explore the action space quickly
            action_p = np.random.uniform(self.lower_bound, self.upper_bound, self.n_actions)
        #clip the resulting action with the bounds
        action_p = np.clip(action_p, self.lower_bound, self.upper_bound)
        return action_p


    def learn(self, step):
        """
        Fill the buffer up to the batch size, then train both networks with
        experience from the replay buffer.
        """
        if self._memory.isReady(self.batch_size):
            self.train_helper(step)

    """
    Train helper methods
    train_helper
    train_critic
    train_actor
    get_q_targets  Q values to train the critic
    get_gradients  policy gradients to train the actor
    """

    def train_helper(self, step):
        # get experience batch
        states, actions, rewards, terminal, states_n = self._memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, dtype=tf.float32)
        states_n = tf.convert_to_tensor(states_n)

        # train the critic before the actor
        self.train_critic(states, actions, rewards, terminal, states_n)
        # delayed actor (policy) update
        if (step + 1) % self.actor_update_delay == 0:
            self.train_actor(states)

        #update the target models
        self.critic.update_target()
        # delayed actor (policy) update
        if (step + 1) % self.actor_update_delay == 0:
            self.actor.update_target()

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
        self.actor.train(states, self.critic.model)

    def remember(self, state, state_new, action, reward, terminal):
        """
        replay buffer interfate to the outsize
        """
        self._memory.remember(state, state_new, action, reward, terminal)

    def save(self):
        ext = ".h5"
        self.actor.model.save(self.path_actor + ext)
        self.critic.model.save(self.path_critic + ext)
        self.actor.target_model.save(self.path_actor + "_target" + ext)
        self.critic.target_model.save(self.path_critic + "_target" + ext)

    def load(self):
        ext = ".h5"
        self.actor.model.model = tf.keras.models.load_model(self.path_actor + ext)
        self.critic.model.save = tf.keras.models.load_model(self.path_critic + ext)
        self.actor.target_model = tf.keras.models.load_model(self.path_actor + "_target" + ext)
        self.critic.target_model = tf.keras.models.load_model(self.path_critic + "_target" + ext)
