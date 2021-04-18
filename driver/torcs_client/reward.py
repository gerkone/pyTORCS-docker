import numpy as np

def custom_reward(obs, obs_prev):
    trackPos = np.array(obs['trackPos'])
    speed = np.array(obs['speedX'])
    reward = speed * np.cos(obs['angle']) - np.abs(speed * np.sin(obs['angle'])) - speed * np.abs(trackPos)
    return reward
