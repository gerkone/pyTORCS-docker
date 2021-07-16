import numpy as np
import h5py

from torcs_client.torcs_client import Client
from torcs_client.utils import start_container, reset_torcs

class Simple(object):
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):
        self.action_dims = action_dims
        self.max_dist = 0
        self.laps = 2
        self.dataset_dist = []
        self.dataset_time = []
        self.dataset_sensors = []
        self.dataset_action = []
        self.track = ""
        self.done = True

    def get_action(self, state, i, track):
        """
        Simple proportional feedback controller
        """
        if self.track != track:
            self.track = track
            self.done = False

        if self.done == False:
            if state["distRaced"] > 0:
                dist = state["distRaced"] % state["trackLen"]
                time = state["totalTime"]

                if dist > self.max_dist:
                    self.max_dist = dist
                    self.dataset_time.append(time)
                    self.dataset_dist.append(dist)

                    self.dataset_sensors.append(state["speedX"])
                    self.dataset_sensors.append(state["speedY"])
                    self.dataset_sensors.append(state["speedZ"])
                    self.dataset_sensors.append(state["angle"])
                    self.dataset_sensors.append(state["trackPos"])
                    self.dataset_sensors.extend(state["wheelSpinVel"])
                    self.dataset_sensors.extend(state["track"])

                    self.dataset_action.append(state["steer"])
                    self.dataset_action.append(state["throttle"])

            if state["distRaced"] > state["trackLen"] * self.laps:
                dataset_file = h5py.File("dataset/berniw_clone_{}.h5".format(track.replace("-", "")), "a")
                # telemetry
                dataset_file.create_dataset("dist", data=self.dataset_dist)
                dataset_file.create_dataset("time", data=self.dataset_time)
                # state
                dataset_file.create_dataset("sensors", data=self.dataset_sensors)
                # action
                dataset_file.create_dataset("action", data=self.dataset_action)

                self.done = True
                self.max_dist = 0
                self.dataset_dist = []
                self.dataset_time = []
                self.dataset_sensors = []
                self.dataset_action = []

        action = np.ones(*self.action_dims)
        return action
