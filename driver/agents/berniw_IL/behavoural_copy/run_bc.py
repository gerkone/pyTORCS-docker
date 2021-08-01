import numpy as np
import os

from agents.berniw_BC.network import Agent

class BC:
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):
        self.agent = Agent(save_dir = os.getcwd() + "/driver/agents/berniw_BC/BC_agent", load = True)

    def get_action(self, state, i, track):

        state_array = np.zeros(28)

        state_array[0] = state["speedX"]
        state_array[1] = state["speedY"]
        state_array[2] = state["speedZ"]
        state_array[3] = state["angle"]
        state_array[4] = state["trackPos"]
        state_array[5:9] = state["wheelSpinVel"]
        state_array[9:28] = state["track"]

        action = self.agent.get_action(state_array)

        return action
