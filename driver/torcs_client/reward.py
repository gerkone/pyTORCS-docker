import numpy as np

steering_threshold = 0.1
speed_w = 0.2
damage_w = 0.1
dist_w = 0.5
range_w = 0.0
angle_w = 0.5
steer_w = 0.0

boring_speed = 3

base_reward = 1

# TODO parametric
rangefinder_angles = [-45, -19, -12, -7, -4, -2.5, -1.7, -1, -.5, 0, .5, 1, 1.7, 2.5, 4, 7, 12, 19, 45]

def custom_reward(obs, obs_prev, action, action_prev, cur_step, terminal):

    def on_track_reward(track_pos):
        on_track = np.abs(track_pos) < 1
        if on_track is True:
            return base_reward
        else:
            return -base_reward

    def dist_reward(d, d_old):
        if d > d_old:
            return 1
        return 0

    def speed_reward(speed):
        return 10 ** (speed / 300)

    def damage_reward(d, d_old):
        return np.clip(d - d_old, 0, 100)

    def direction_rangefinder_reward(rangefinder):
        # get all max indices ( in case of long straight there will be multiple 200 m )
        id_max_dist = np.argwhere(rangefinder == np.amax(rangefinder))
        # closest index to center
        id_max_dist = id_max_dist[min(range(len(id_max_dist)), key=lambda i: abs(id_max_dist[i] - 9))][0]
        # reward when car is pointing towards longest distance
        T = np.clip(np.abs(id_max_dist - 9), 0, 4)
        if T <= 1:
            return 5
        else:
            return -T

    def direction_angle_reward(angle):
        a = np.cos(angle)
        if a <= 0:
            return -50
        return np.clip(2 ** (1 / a), 0, 1)

    def straight_line_reward(steering, speed):
        if np.abs(steering) < steering_threshold and speed > boring_speed:
            return 1
        return 0

    def wobbly_reward(steering, steering_old):
        if np.abs(np.abs(steering) - np.abs(steering_old)) > steering_threshold:
            return -0.1
        return 0

    def breaking_reward(speed, brake):
        if speed <= boring_speed and brake > 0:
            return -0.1
        return 0

    # punish for standing still
    if obs["speedX"] > boring_speed:
        # basic reward
        reward = on_track_reward(obs["trackPos"])
        # speed reward
        reward += speed_reward(obs["speedX"]) * speed_w
        # travel distance reward
        reward += dist_reward(obs["distRaced"], obs_prev["distRaced"]) * dist_w
        # punishment for damage
        reward += damage_reward(obs["damage"], obs_prev["damage"]) * damage_w
        # direction dependent rewards
        reward += direction_rangefinder_reward(obs["track"]) * range_w
        reward += direction_angle_reward(obs["angle"]) * angle_w
        # reward going straight, punish wobbling
        reward += straight_line_reward(action["steer"], obs["speedX"]) * steer_w
        reward += wobbly_reward(action["steer"], action_prev["steer"])
        # punish breaking when going slow
        reward += breaking_reward(action["brake"], obs["speedX"])
    else:
        reward = -1

    return reward
