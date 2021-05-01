import numpy as np

from torcs_client.torcs_client import Client
from torcs_client.utils import start_container, reset_torcs

class Simple(object):
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):
        # normalized target speed
        self.target_speed = hyperparams["target_speed"]
        self.norm_factor = hyperparams["max_speed"]
        self.noise_scale = hyperparams["noise_scale"]
        self.frame_freq = hyperparams["frame_freq"]

        self.action_dims = action_dims

        self.episode_dataset = np.empty(shape = 0, dtype = object)

        self.prev_accel = 0

        self.curr_step = 0

    def get_action(self, state, i):
        """
        Simple proportional feedback controller
        """

        speedX = state["speedX"] * self.norm_factor

        action = np.zeros(*self.action_dims)
        # steer to corner
        steer = state["angle"] * 10
        # steer to center
        steer -= state["trackPos"] * .05

        accel = self.prev_accel

        if speedX < self.target_speed - (steer * 50):
            accel += .05
        else:
            accel -= .05

        if accel > 0.4:
            accel = 0.4

        if speedX < 10:
            accel += 1 / (speedX + .1)

        self.prev_accel = accel

        action[0] = steer
        action[1] = accel

        noise = np.random.uniform(low = -1, high = 1, size = 2)

        noise_scaled = noise * self.noise_scale

        action_noised = action + noise_scaled

        self.curr_step += 1
        self.curr_episode = i

        if self.curr_step % self.frame_freq == 0:
            self.store_state(state)


        return action_noised

    def store_state(self, state):
        img = np.asarray(state["img"])
        del state["img"]
        state = np.array(list(state.values()), dtype = object)
        state = np.append(state, img)

        self.episode_dataset = np.append(self.episode_dataset, state)

    def save_models(self):
        self.curr_step = 0
        with open("dataset/ep_{}.npy".format(self.curr_episode), "wb") as f:
            np.save(f, self.episode_dataset)
        self.episode_dataset = np.empty(shape = 0, dtype = object)
