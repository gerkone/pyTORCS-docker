import numpy as np
import h5py
import os
import time
from math import floor

class BaseReward():
    def __init__(self):
        self.track = ""
        self.base_reward = 1

    def _damage_reward(self, d, d_old):
        return -np.clip(d - d_old, 0, 100)

    def _on_track_reward(self, track_pos):
        on_track = np.abs(track_pos) < 1
        if on_track is True:
            return self.base_reward
        else:
            return -self.base_reward

    def get_reward(self, obs, obs_prev, action, action_prev, cur_step, terminal, track):
        return self.base_reward

    def reset(self):
        pass

######## Local reward ########
class LocalReward(BaseReward):
    steering_threshold = 0.1
    base_w = 0.0
    speed_w = 0.0
    damage_w = 0.1
    dist_w = 1.0
    range_w = 0.0
    angle_w = 0.1
    steer_w = 0.0

    wobble_w = 0.0

    boring_speed = 0.5

    # TODO parametric
    rangefinder_angles = [-45, -19, -12, -7, -4, -2.5, -1.7, -1, -.5, 0, .5, 1, 1.7, 2.5, 4, 7, 12, 19, 45]

    def __dist_reward(self, d, d_old):
        return np.clip(d - d_old, -1, 5)

    def __speed_reward(self, speed):
        return 10 ** (speed / 300)

    def __direction_rangefinder_reward(self, rangefinder):
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

    def __direction_angle_reward(self, angle):
        a = np.cos(angle)
        if a <= 0:
            return -50
        return np.clip(a, 0, 1)

    def __straight_line_reward(self, steering, speed):
        if np.abs(steering) < self.steering_threshold and speed > self.boring_speed:
            return 1
        return 0

    def __wobbly_reward(self, steering, steering_old):
        if np.abs(np.abs(steering) - np.abs(steering_old)) > self.steering_threshold:
            return -0.1
        return 0

    def __breaking_reward(self, speed, brake):
        if speed <= self.boring_speed and brake > 0:
            return -0.1
        return 0

    def __local_reward(self, obs, obs_prev, action, action_prev, cur_step, terminal):
        # punish for standing still
        try:
            # basic reward
            reward = self._on_track_reward(obs["trackPos"]) * self.base_w
        except Exception:
            reward = 0
        try:
            # speed reward
            reward += self.__speed_reward(obs["speedX"]) * self.speed_w
        except Exception:
            pass
        try:
            reward += self.__dist_reward(obs["distRaced"], obs_prev["distRaced"]) * self.dist_w
            # travel distance reward
        except Exception:
            pass
        try:
            # punishment for damage
            reward += self._damage_reward(obs["damage"], obs_prev["damage"]) * self.damage_w
        except Exception:
            pass
        try:
            # direction dependent rewards
            reward += self.__direction_rangefinder_reward(obs["track"]) * self.range_w
        except Exception:
            pass
        try:
            reward += self.__direction_angle_reward(obs["angle"]) * self.angle_w
        except Exception:
            pass
        try:
            # reward going straight, punish wobbling
            reward += self.__straight_line_reward(action["steer"], obs["speedX"]) * self.steer_w
        except Exception:
            pass
        try:
            reward += self.__wobbly_reward(action["steer"], action_prev["steer"]) * self.wobble_w
        except Exception:
            pass
        if obs["speedX"] > self.boring_speed:
            reward -= 1
        try:
            # punish breaking when going slow
            reward += self.__breaking_reward(action["brake"], obs["speedX"])
        except Exception:
            pass
        return reward

    # overridden
    def get_reward(self, obs, obs_prev, action, action_prev, cur_step, terminal, track):
        return self.__local_reward(obs, obs_prev, action, action_prev, cur_step, terminal)


######## General time dependent reward, berniw as reference ########
class TimeReward(BaseReward):
    damage_w = 5.0
    reference_w = 1.0
    sector_length = 10.0

    def __find_nearest(self, a, v):
        idx = np.searchsorted(a, v, side = "left")
        if idx > 0 and (idx == len(a) or np.abs(v - a[idx-1]) < np.abs(v - a[idx])):
            return idx-1
        else:
            return idx

    def __load_dataset(self):
        try:
            dataset = h5py.File(os.path.join(os.getcwd(), "driver/torcs_client/reward_dataset/berniw_timemap_{}.h5".format(self.track.replace("-", ""))), "r+")
            self.times = np.array(dataset.get("time"))
            self.distances = np.array(dataset.get("dist"))
        except Exception:
            self.times = None
            self.distances = None

    def __temporal_competing_reward(self, obs, obs_prev, terminal):
        if self.starttime_sector == -1:
            self.starttime_sector = obs["totalTime"]

        # if past last sector give reward
        if floor(obs["distRaced"] / self.sector_length) * self.sector_length > self.curr_sector:
            self.curr_sector = round(obs["distRaced"] / self.sector_length) * self.sector_length
            self.prev_sector = 0 if self.curr_sector == 0 else self.curr_sector - self.sector_length

            nearest_idx_1 = self.__find_nearest(self.distances, self.curr_sector)
            nearest_idx_2 = self.__find_nearest(self.distances, self.prev_sector)

            # reference sector time
            reference = self.times[nearest_idx_2] - self.times[nearest_idx_1]

            # agent sector time
            sector_time = obs["totalTime"] - self.starttime_sector

            reward = (reference - sector_time) * self.reference_w

            # start time for the next sector
            self.starttime_sector = obs["totalTime"]
        else:
            # if still in sector nothing
            reward = 0

        reward += self._damage_reward(obs["damage"], obs_prev["damage"]) * self.damage_w

        # punish for not moving forward
        if terminal and self.curr_sector == 0:
            reward = -1000
        # punish for spins
        if np.cos(obs["angle"]) < 0:
            reward = -100
        return reward

    # overridden
    def get_reward(self, obs, obs_prev, action, action_prev, cur_step, terminal, track):
        if track != self.track:
            self.track = track
            # load track time map
            self.__load_dataset()
        return self.__temporal_competing_reward(obs, obs_prev, terminal)

    def reset(self):
        self.curr_sector = 0
        self.prev_sector = 0
        self.starttime_sector = -1
