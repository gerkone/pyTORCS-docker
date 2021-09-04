import numpy as np
import h5py

from torcs_client.torcs_client import Client
from torcs_client.utils import start_container, reset_torcs

class Simple(object):
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):
        self.action_dims = action_dims
        self.max_dist = 0
        self.laps = 1
        self.track = ""
        self.dataset_dist = []
        self.dataset_time = []
        self.dataset_sensors = []
        self.dataset_sensors_new = []
        self.dataset_action = []
        self.dataset_reward = []

    def get_action(self, state, i, track):
        """
        Simple proportional feedback controller
        """
        if self.track != track:
            self.track = track
        action = np.ones(*self.action_dims)
        return action

    def remember(self, state, state_new, action, reward, terminal):
        time = state["totalTime"]
        self.dataset_time.append(time)

        dist = state["distRaced"] % state["trackLen"]
        self.dataset_dist.append(dist)

        state_array = np.zeros(28)

        state_array[0] = state["speedX"]
        state_array[1] = state["speedY"]
        state_array[2] = state["speedZ"]
        state_array[3] = state["angle"]
        state_array[4] = state["trackPos"]
        state_array[5:9] = state["wheelSpinVel"]
        state_array[9:28] = state["track"]

        self.dataset_sensors.append(state_array)

        state_array = np.zeros(28)

        state_array[0] = state_new["speedX"]
        state_array[1] = state_new["speedY"]
        state_array[2] = state_new["speedZ"]
        state_array[3] = state_new["angle"]
        state_array[4] = state_new["trackPos"]
        state_array[5:9] = state_new["wheelSpinVel"]
        state_array[9:28] = state_new["track"]

        self.dataset_sensors_new.append(state_array)

        action = np.ones(2)

        action[0] = state["steer"]
        action[1] = state["throttle"]

        self.dataset_action.append(action)

        self.dataset_reward.append(reward)

        if terminal == True:
            # remember done
            dataset_file = h5py.File("dataset/berniw_clone_{}.h5".format(self.track.replace("-", "")), "a")
            # telemetry
            dataset_file.create_dataset("dist", data = self.dataset_dist)
            dataset_file.create_dataset("time", data = self.dataset_time)
            # state
            dataset_file.create_dataset("sensors", data = self.dataset_sensors)
            dataset_file.create_dataset("sensors_new", data = self.dataset_sensors_new)
            # action
            dataset_file.create_dataset("action", data = self.dataset_action)
            # reward
            dataset_file.create_dataset("reward", data = self.dataset_reward)

            self.dataset_dist = []
            self.dataset_time = []
            self.dataset_sensors = []
            self.dataset_action = []
