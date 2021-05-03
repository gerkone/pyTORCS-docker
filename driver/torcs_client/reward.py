import numpy as np

max_steer_frame = 0.2
damage_w = 0.3
dist_w = 0.9

def custom_reward(obs, obs_prev, action, action_prev):
    # basic reward
    reward = (obs["distRaced"] - obs_prev["distRaced"]) * dist_w
    if obs["speedX"] < 0:
        reward -= .1
        if action["brake"] > 0:
            reward -= .5

    if np.cos(obs["angle"]) < 0:
        reward -= 5

    # punishment for damage
    reward -= np.clip(obs["damage"] - obs_prev["damage"], 0, 100) * damage_w

    # # punishment for too much steering
    # if action["steer"] > 0.5:
    #     reward -= 0.05
    #
    # punishment for wobbly steering
    if action["steer"] > action_prev["steer"] + max_steer_frame or action["steer"] < action_prev["steer"] - max_steer_frame:
        reward -= 0.5

    return reward
