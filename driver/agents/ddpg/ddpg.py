import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

import os
import time
import h5py

from agents.ddpg.utils.replay_buffer import ReplayBuffer
from agents.ddpg.utils.action_noise import OUActionNoise
from agents.ddpg.network.actor import Actor
from agents.ddpg.network.critic import Critic

from torcs_client.reward import LocalReward
from torcs_client.utils import SimpleLogger as log

class DDPG(object):
    """
    DDPG agent
    """
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):

        # physical_devices = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)

        actor_lr = hyperparams["actor_lr"]
        critic_lr = hyperparams["critic_lr"]
        batch_size = hyperparams["batch_size"]
        gamma = hyperparams["gamma"]
        self.buf_size = int(hyperparams["buf_size"])
        tau = hyperparams["tau"]
        fcl1_size = hyperparams["fcl1_size"]
        fcl2_size = hyperparams["fcl2_size"]
        guided_episode = hyperparams["guided_episode"] - 1
        save_dir = hyperparams["save_dir"]

        noise_phi = hyperparams["noise_phi"]

        # action size
        self.n_states = state_dims[0]
        # state size
        self.n_actions = action_dims[0]
        self.batch_size = batch_size

        self.guided_episode = guided_episode
        self.save_dir = save_dir

        # environmental action boundaries
        self.lower_bound = action_boundaries[0]
        self.upper_bound = action_boundaries[1]

        self.noise_phi = noise_phi

        # experience replay buffer
        self._memory = ReplayBuffer(self.buf_size, input_shape = state_dims, output_shape = action_dims)
        # noise generator
        self._noise = OUActionNoise(mu=np.zeros(action_dims))
        # Bellman discount factor
        self.gamma = gamma

        self.prev_accel = 0

        self.track = ""

        # turn off most logging
        logging.getLogger("tensorflow").setLevel(logging.FATAL)

        # date = datetime.now().strftime("%m%d%Y_%H%M%S")
        # path_actor = "./models/actor/actor" + date + ".h5"
        # path_critic = "./models/critic/actor" + date + ".h5"

        # actor class
        self.actor = Actor(state_dims = state_dims, action_dims = action_dims,
                lr = actor_lr, batch_size = batch_size, tau = tau,
                upper_bound = self.upper_bound, save_dir = self.save_dir,
                fcl1_size = fcl1_size, fcl2_size = fcl2_size)
        # critic class
        self.critic = Critic(state_dims = state_dims, action_dims = action_dims,
                lr = critic_lr, batch_size = batch_size, tau = tau,
                save_dir = self.save_dir, fcl1_size = fcl1_size, fcl2_size = fcl2_size)

        # pretrain networks on expert data
        if hyperparams["pretrain"] == True:
            # initialize virtual reward
            self.rewarder = LocalReward()
            self.pretrain(hyperparams["epochs"], hyperparams["dataset_dir"])

    def repack_state(self, state_array):
        """
        state array to state dict
        """
        state = {}

        state["speedX"] = state_array[0]
        state["speedY"] = state_array[1]
        state["speedZ"] = state_array[2]
        state["angle"] = state_array[3]
        state["trackPos"] = state_array[4]
        state["wheelSpinVel"] = state_array[5:9]
        state["track"] = state_array[9:28]

        return state

    def repack_action(self, action_array):
        """
        state array to state dict
        """
        action = {}

        action["steer"] = action_array[0]
        action["throttle"] = np.clip(action_array[1], 0, 1)
        action["brake"] = np.clip(-action_array[1], 0, 1)

        return action

    def get_action(self, state, episode, track):
        """
        Return the best action in the passed state, according to the model
        in training. Noise added for exploration
        """
        if self.track != track:
            self.track = track
        #take only random actions for the first episode
        if(episode > self.guided_episode):
            state = self._memory.unpack_state(state)
            state = tf.expand_dims(state, axis = 0)
            action = self.actor.model.predict(state)[0]
        else:
            #explore the action space quickly
            action = self.simple_controller(state)

        noise = self._noise()

        if np.random.random() > self.noise_phi:
            # set noise on steer to 0 for some steps
            # this helps throttle exploration
            noise[1] = noise[0] + noise[1]
            noise[0] = 0

        action_p = action + noise

        #clip the resulting action with the bounds
        action_p = np.clip(action_p, self.lower_bound, self.upper_bound)
        return action_p

    def _prepare_data(self, filename):
        dataset = h5py.File(filename, "r")

        action = np.array(dataset.get("action"))
        sensors = np.array(dataset.get("sensors"))
        distances = np.array(dataset.get("dist"))

        return action, sensors, distances

    def pretrain(self, epochs, dataset_dir):
        dataset_files = []

        for file in os.listdir(dataset_dir):
            if ".h5" in file:
                dataset_files.append(os.path.join(dataset_dir, file))

        curr_ep = 0
        # used tu resume the dataset if not completed
        last_el = 0

        while curr_ep < len(dataset_files):
            ep_file = dataset_files[curr_ep]

            log.info("Loading expert trjectory: {} - {}/{}".format(ep_file, curr_ep + 1, len(dataset_files)))

            tot = 0
            action, sensors, distances = self._prepare_data(ep_file)

            for el in range(last_el, len(action) - 1):
                if not self._memory.isFull():
                    # fill replay buffer
                    prev = el - 1 if el > 0 else 0
                    # calculate what the reward would be
                    repacked_state_curr = self.repack_state(sensors[el])
                    repacked_state_prev = self.repack_state(sensors[prev])
                    repacked_state_curr["distRaced"] = distances[el]
                    repacked_state_prev["distRaced"] = distances[prev]
                    repacked_state_curr["damage"] = 0
                    repacked_state_prev["damage"] = 0
                    repacked_action_curr = self.repack_action(action[el])
                    repacked_action_prev = self.repack_action(action[prev])
                    reward = self.rewarder.get_reward(repacked_state_curr, repacked_state_prev, repacked_action_curr, repacked_action_prev, el, terminal = False, track = self.track)
                    tot += reward
                    self.remember(sensors[el], sensors[el + 1], action[el], reward, 0)
                else:
                    last_el = el

            if self._memory.isFull():
                time_start = time.time()
                for e in range(epochs):
                    loss = self.learn(0)
                    log.training("Epoch {}. ".format(e + 1), loss)
                time_end = time.time()
                log.info("Completed {:d} epochs. Duration {:.2f} ms".format(epochs, 1000.0 * (time_end - time_start)))
                # empty the buffer
                self._memory.reset()
                last_el = 0

            if last_el == 0:
                # next file only if current was completed
                curr_ep += 1

            del action
            del sensors

        log.info("Pretraining complete. Saving models...")
        self.save_models()

    def simple_controller(self, state):
        speedX = state["speedX"]
        action = np.zeros(self.n_actions)
        # steer to corner
        steer = state["angle"] * 12
        # # steer to center
        steer -= state["trackPos"] * .2

        accel = self.prev_accel

        if speedX < 120 - (steer * 50):
            accel += .03
        else:
            accel -= .03

        if accel > 0.5:
            accel = 0.5

        if speedX < 10:
            accel += 1 / (speedX + .1)


        self.prev_accel = accel

        action[0] = steer
        action[1] = accel

        self.prev_accel = accel

        return action


    def learn(self, i, trajectory_mode = True):
        """
        Fill the buffer up to the batch size, then train both networks with
        experience from the replay buffer.
        """
        avg_loss = 0
        if self._memory.isReady(self.batch_size):
            for r in range(int(len(self._memory) / self.batch_size)):
                avg_loss += self.train_helper(trajectory_mode)

        return avg_loss / (len(self._memory) / self.batch_size)

    def save_models(self):
        self.actor.model.save(self.save_dir + "/actor")
        self.actor.target_model.save(self.save_dir + "/actor_target")
        self.critic.model.save(self.save_dir + "/critic")
        self.critic.target_model.save(self.save_dir + "/critic_target")

    """
    Train helper methods
    train_helper
    train_critic
    train_actor
    get_q_targets  Q values to train the critic
    get_gradients  policy gradients to train the actor
    """

    def train_helper(self, trajectory_mode):
        # get experience batch
        states, actions, rewards, terminal, states_n = self._memory.sample(self.batch_size, trajectory_mode)
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
