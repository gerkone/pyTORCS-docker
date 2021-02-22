import numpy as np

def custom_reward(obs, obs_prev):
    # direction-dependent positive reward
    sp = np.array(obs['speedX'])
    reward = sp*np.cos(obs['angle'])

    # future better reward
    # trackPos = np.array(obs['trackPos'])
    # speed = np.array(obs['speedX'])
    # reward = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - speed * np.abs(trackPos)

    return reward
