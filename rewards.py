import numpy as np

def custom_reward(obs, obs_prev):
    # direction-dependent positive reward

    sp = np.array(obs['speedX'])

    progress = sp*np.cos(obs['angle'])
    reward = progress

    # collision detection
    if obs['damage'] - obs_prev['damage'] > 0:
        reward = -1

    return reward


def better_reward(obs, obs_prev):
    trackPos = np.array(obs['trackPos'])
    speed = np.array(obs['speedX'])

    progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - speed * np.abs(trackPos)
    reward = progress

    # collision detection
    if obs['damage'] - obs_pre['damage'] > 0:
        reward = -1
