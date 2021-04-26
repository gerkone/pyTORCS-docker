import numpy as np


def custom_reward(obs, obs_prev):
    reward = obs["distFromStart"] - obs_prev["distFromStart"]
    if obs["speedX"] < 0:
        reward -= 100
    if np.cos(obs["angle"]) < 0:
        reward -= 100
    return reward
