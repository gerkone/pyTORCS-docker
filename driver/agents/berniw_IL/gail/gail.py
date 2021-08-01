import numpy as np
import tensorflow as tf
import os
import h5py
import copy

from agents.berniw_IL.gail.replay_buffer import ReplayBuffer

from agents.berniw_IL.gail.network.discriminator import Discriminator
from agents.berniw_IL.gail.network.policy import Policy

save_dir = "driver/agents/berniw_IL/gail/model"

dataset_dir = "driver/agents/berniw_IL/dataset"

class GAIL:
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):

        self.state_dims = state_dims
        self.action_dims = [len(hyperparams["action_map"])]

        self.dataset_dir = dataset_dir

        self.batch_size = hyperparams["batch_size"]
        self.lr = hyperparams["lr"]
        self.gamma = hyperparams["gamma"]
        self.buf_size = hyperparams["buf_size"]
        self.labmda = hyperparams["labmda"]
        self.action_map = hyperparams["action_map"]

        self.track = ""
        self.episode = 0

        self.d = Discriminator(state_dims = self.state_dims, action_dims = self.action_dims, batch_size = self.batch_size, lr = self.lr, load_dir = save_dir)
        self.pi = Policy(state_dims = self.state_dims, action_dims = self.action_dims, batch_size = self.batch_size, gamma = self.gamma, lr = self.lr, load_dir = save_dir)

        self._memory = ReplayBuffer(self.buf_size, input_shape = state_dims, output_shape = action_dims)

        self.v_preds = []

    def discretize(self, action_cont):
        # find closest discrete action
        action_disc = np.zeros(shape = (action_cont.shape[0], *self.action_dims))

        for i in range(action_cont.shape[0]):
            on = min(range(*self.action_dims), key = lambda e : ((self.action_map[e] - action_cont[i])**2).mean(axis=0))
            action_disc[i][on] = 1

        return action_disc

    def load_expert(self, track):
        dataset = h5py.File("{}/berniw_clone_{}.h5".format(self.dataset_dir, track.replace("-","")), "r")

        action_cont = np.array(dataset.get("action"))
        action_disc = self.discretize(action_cont)

        sensors = np.array(dataset.get("sensors"))

        return [sensors, action_disc]

    def unpack_state(self, state):
        state_array = np.zeros(self.state_dims)

        state_array[0] = state["speedX"]
        state_array[1] = state["speedY"]
        state_array[2] = state["speedZ"]
        state_array[3] = state["angle"]
        state_array[4] = state["trackPos"]
        state_array[5:9] = state["wheelSpinVel"]
        state_array[9:28] = state["track"]

        return state_array

    def get_action(self, state, i, track):
        if self.episode != i:
            # reset buffers
            self.reset()
            self.episode = i

        if self.track != track:
            self.track = track
            self.expert = self.load_expert(self.track)

        state_array = self.unpack_state(state)

        action = self.pi.get_action(state_array)
        v_pred = self.pi.get_value(state_array)

        self.v_preds.append(v_pred)

        return self.action_map[np.argmax(action)]

    def remember(self, state, state_new, action, reward, terminal):
        self._memory.remember(state, state_new, action, reward, terminal)

        if terminal == True:
            state_array = self.unpack_state(state)
            last_v_pred = self.pi.get_value(state_array)
            self.v_preds = np.array(self.v_preds)
            self.v_preds_next = np.append(self.v_preds[1:], np.asscalar(last_v_pred)).astype(dtype = np.float32)

    def learn(self, i):
        cur = 0
        loss = 0
        while not self._memory.done:

            state, action, _, _, _ = self._memory.sample(self.batch_size)

            action = self.discretize(action)

            ###### train discriminator ######
            policy_s = tf.convert_to_tensor(state)
            policy_a = tf.convert_to_tensor(action)

            batch = np.arange(cur, cur + self.batch_size, 1)

            expert_s = self.expert[0][batch]
            expert_a = self.expert[1][batch]

            self.d.train(expert_s, expert_a, policy_s, policy_a)

            cur += self.batch_size

            # estimated reward by discriminator on batch
            estimated_rewards = self.d.get_reward(policy_s, policy_a)

            ###### train policy with ppo ######
            gae = self.get_gae(estimated_rewards, self.v_preds, self.v_preds_next)

            loss = self.pi.train(policy_s, policy_a, gae, estimated_rewards, self.v_preds_next)

        # ready for next epoch
        self._memory.done = False

        return loss


    def get_gae(self, rewards, v_preds, v_preds_next):
        """
        generative advantage estimator
        """
        deltas = []
        for i in reversed(range(rewards.shape[0])):
            V = rewards[i] + self.gamma * v_preds_next[i]
            delta = V - v_preds[i]
            deltas.append(delta)
        deltas = np.array(list(reversed(deltas)))
        # compute gae
        A = deltas[-1,:]
        advantages = [A]
        for i in reversed(range(deltas.shape[0] - 1)):
            A = deltas[i] + self.gamma * v_preds_next[i] * A * self.labmda
            advantages.append(A)
        advantages = reversed(advantages)
        # (T, N) -> (N, T)
        advantages = np.transpose(list(advantages), [1, 0])
        return advantages


    def reset(self):
        self.v_preds = []
        self._memory.reset()
