import numpy as np

from torcs_client.torcs_client import Client
from torcs_client.utils import start_container, reset_torcs

class Simple(object):
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):
        # normalized target speed
        self.target_speed = hyperparams["target_speed"]
        self.action_dims = action_dims

        self.prev_accel = 0

    def get_action(self, state, i, track):
        """
        Simple proportional feedback controller
        """
        speedX = state["speedX"]

        action = np.zeros(*self.action_dims)
        # steer to corner
        steer = state["angle"] * 10
        # steer to center
        steer -= state["trackPos"] * .10

        accel = self.prev_accel


        if speedX < self.target_speed - (steer * 50):
            accel += .01
        else:
            accel -= .01

        if accel > 0.2:
            accel = 0.2

        if speedX < 10:
            accel += 1 / (speedX + .1)

        # traction control system
        if ((state["wheelSpinVel"][2]+state["wheelSpinVel"][3]) -
           (state["wheelSpinVel"][0]+state["wheelSpinVel"][1]) > 5):
            accel -= .2

        self.prev_accel = accel

        action[0] = steer
        action[1] = accel

        return action
