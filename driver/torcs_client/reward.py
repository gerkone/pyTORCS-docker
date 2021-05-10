import numpy as np

max_steer_frame = 0.2
damage_w = 1
dist_w = 0.9

reward_rate = 50


def custom_reward(obs, obs_prev, action, action_prev, cur_step, terminal):
    # basic reward
    distance = obs["distRaced"] - obs_prev["distRaced"]
    reward = distance * dist_w

    # punishment for damage
    if obs["damage"] > obs_prev["damage"]:
        reward = -10

    if np.cos(obs["angle"]) < 0:
        reward = -50

    return reward
