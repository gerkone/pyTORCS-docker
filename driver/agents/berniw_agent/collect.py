import numpy as np
import h5py

from torcs_client.torcs_client import Client
from torcs_client.utils import start_container, reset_torcs

class Simple(object):
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):
        self.action_dims = action_dims
        self.max_dist = 0
        self.dataset_dist = []
        self.dataset_time = []

    def get_action(self, state, i, track):
        """
        Simple proportional feedback controller
        """
        print(state)
        if state["distraced"] > 0:
            dist = state["distraced"] % state["tracklen"]
            time = state["curtime"]
            if dist > self.max_dist:
                self.max_dist = dist
                self.dataset_time.append(time)
                self.dataset_dist.append(dist)

        if state["distraced"] > state["tracklen"]:
            dataset_file = h5py.File("dataset/berniw_timemap_{}.h5".format(track.replace("-", "")), "a")
            dataset_file.create_dataset("dist", data=self.dataset_dist)
            dataset_file.create_dataset("time", data=self.dataset_time)
            input("Done!")
            exit()

        action = np.ones(*self.action_dims)
        return action
