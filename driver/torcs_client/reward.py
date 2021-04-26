import numpy as np

def custom_reward(obs, obs_prev, action):
    reward = obs["distRaced"] - obs_prev["distRaced"]
    if obs["speedX"] < 0:
        reward -= .1
        if action["brake"] > 0:
            reward -= .5

    if np.cos(obs["angle"]) < 0:
        reward -= 5
    return reward
