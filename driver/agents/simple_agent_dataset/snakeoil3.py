import numpy as np
import h5py

from torcs_client.torcs_client import Client
from torcs_client.utils import start_container, reset_torcs

class Simple(object):
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):
        # normalized target speed
        self.target_speed = hyperparams["target_speed"]
        self.noise_scale = hyperparams["noise_scale"]
        self.state_dims = state_dims
        self.action_dims = action_dims

        # noise = np.random.uniform(low = - self.target_speed / 3, high = self.target_speed / 3)
        # self.target_speed += noise
        self.target_speed = np.clip(self.target_speed, 0, 300)

        self.episode = 0
        self.total_episodes = 0
        self.first_step = True
        self.passed = False

        self.prev_accel = 0

    def get_action(self, state, i, track):
        """
        Simple proportional feedback controller
        """
        if self.episode < i or (i == 0 and self.total_episodes > 0 and self.episode != 0):
            self.episode = i
            self.total_episodes += 1
            self.first_step = True

        speedX = state["speedX"]

        action = np.zeros(*self.action_dims)
        # steer to corner
        steer = state["angle"] * 12
        # steer to center
        steer -= state["trackPos"] * .2

        # if speedX < 150 and self.passed == False:
        #     accel = 0.3
        # elif speedX > 150:
        #     self.passed = True
        # elif speedX < 20:
        #     self.passed == False
        #     accel = 0.3
        # else:
        #     accel = -0.3

        accel = self.prev_accel

        if speedX < self.target_speed - (steer * 50):
            accel += .03
        else:
            accel -= .03

        if accel > 0.3:
            accel = 0.3

        if speedX < 10:
            accel += 1 / (speedX + .1)

        self.prev_accel = accel

        action[0] = steer
        action[1] = accel

        noise = np.random.uniform(low = -1, high = 1, size = 2)

        noise_scaled = noise * self.noise_scale

        action_noised = action + noise_scaled

        self.store_state(state, track)

        return action_noised

    def store_state(self, state, track):
        if "img" in state.keys():
            img = np.asarray(state["img"], dtype = np.uint8)
            img = np.expand_dims(img, axis = 0)
            del state["img"]
            sensors = np.hstack(list(state.values()))
            sensors = np.expand_dims(sensors, axis = 0)
            if self.first_step:
                self.first_step = False
                self.dataset_file = h5py.File("dataset/ep_{}_{}.h5".format(self.total_episodes, track.replace("-", "")), "a")
                self.dataset_file.create_dataset("img", data=img, compression="gzip", chunks=True, maxshape=(None, img.shape[1], img.shape[2], img.shape[3]))
                self.dataset_file.create_dataset("sensors", data=sensors, compression="gzip", chunks=True, maxshape=(None, *self.state_dims))
            else:
                self.dataset_file["img"].resize((self.dataset_file["img"].shape[0] + img.shape[0]), axis = 0)
                self.dataset_file["img"][-img.shape[0]:] = img
                self.dataset_file["sensors"].resize((self.dataset_file["sensors"].shape[0] + sensors.shape[0]), axis = 0)
                self.dataset_file["sensors"][-sensors.shape[0]:] = sensors
