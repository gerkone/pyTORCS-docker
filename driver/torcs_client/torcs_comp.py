from gym import spaces
import numpy as np
# from os import path
import copy
import collections as col
import matplotlib.pyplot as plt
import sys, signal

from torcs_client.torcs_client import Client
from torcs_client.reward import custom_reward
from torcs_client.terminator import custom_terminal
from torcs_client.utils import start_container, reset_torcs, kill_torcs

class TorcsEnv:
    def __init__(self, throttle = False, gear_change = False, state_filter = None, default_speed = 50,
            port = 3001, img_width = 640, img_height = 480, verbose = False, image_name = "gerkone/torcs"):

        self.throttle = throttle
        self.gear_change = gear_change
        self.default_speed = default_speed

        self.verbose = verbose

        self.image_name = image_name

        if self.image_name != "0":
            # start torcs container
            self.container_id = start_container(self.image_name, self.verbose, port)
        else:
            self.container_id = "0"

        # should be true if tor was just opened
        self.initial_run = True

        if img_width != 0:
            self.img_width = img_width
        else:
            self.img_width = 64

        if img_height != 0:
            self.img_height = img_height
        else:
            self.img_height = 64

        if state_filter != None:
            self.state_filter = dict(sorted(state_filter.items()))
        else:
            self.state_filter = {}
            self.state_filter["angle"] = np.pi
            self.state_filter["track"] = 200.0
            self.state_filter["trackPos"] = 1.0
            self.state_filter["speedX"] = 300.0
            self.state_filter["speedY"] = 300.0
            self.state_filter["speedZ"] = 300.0
            self.state_filter["wheelSpinVel"] = 1.0
            self.state_filter["rpm"] = 10000

        vision = "img" in state_filter

        # run torcs and start practice run
        reset_torcs(self.container_id, vision, True)

        # create new torcs client - after torcs launch
        self.client = Client(port = port, verbose = self.verbose, container_id = self.container_id,
                        maxSteps = np.inf, vision = vision)

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        high = np.array([])
        low = np.array([])
        if "angle" in self.state_filter:
            high = np.append(high, 1.0)
            low = np.append(low, -1.0)
        if "rpm" in self.state_filter:
            high = np.append(high, np.inf)
            low = np.append(low, 0.0)
        if "speedX" in self.state_filter:
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "speedY" in self.state_filter:
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "speedZ" in self.state_filter:
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "track" in self.state_filter:
            # the track rangefinder is made of 19 separate values
            high = np.append(high, np.ones(19))
            low = np.append(low, np.zeros(19))
        if "trackPos" in self.state_filter:
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "wheelSpinVel" in self.state_filter:
            # one value each wheel
            high = np.append(high, np.array([np.inf, np.inf, np.inf, np.inf]))
            low = np.append(low, np.zeros(4))

        self.observation_space = spaces.Box(low=low, high=high)

        # kill torcs on sigint, avoid leaving the open window
        def kill_torcs_and_close(sig, frame):
            kill_torcs(self.container_id)
            sys.exit(0)

        signal.signal(signal.SIGINT, kill_torcs_and_close)

    def step(self, u):
        # get the state from torcs - simulation step
        self.client.get_servers_input()

        # convert u to the actual torcs actionstr
        action = self.agent_to_torcs(u)

        # current observation
        curr_state = self.client.S.d

        # Steering
        self.client.R.d["steer"] = action["steer"]  # in [-1, 1]

        # Simple Automatic Throttle Control by Snakeoil
        if self.throttle is False:
            self.client.R.d["accel"] = self.automatic_throttle_control(self.default_speed, curr_state, self.client.R.d["accel"], self.client.R.d["steer"])
        else:
            self.client.R.d["accel"] = action["accel"]
            self.client.R.d["brake"] = action["brake"]

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is False:
            self.client.R.d["gear"] = self.automatic_gearbox(curr_state["speedX"], self.client.R.d["gear"])
        else:
            self.client.R.d["gear"] = action["gear"]

        # Save the privious full-obs from torcs for the reward calculation
        obs_prev = copy.deepcopy(curr_state)

        # Apply the agent"s action into torcs
        self.client.respond_to_server()

        # Get the current full-observation from torcs
        obs = curr_state

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # ################### Reward setting Here ###################
        reward = custom_reward(obs, obs_prev)
        # ################### Termination judgement ###################
        episode_terminate = custom_terminal(obs, reward, time_step = self.time_step)

        # reset torcs on terminate
        self.client.R.d["meta"] = episode_terminate

        if episode_terminate:
            self.initial_run = False

        self.time_step += 1

        return self.observation, reward, episode_terminate

    def reset(self):
        if self.verbose: print("Reset")

        self.time_step = 0

        # Get the initial full-observation from torcs
        obs = self.client.S.d


        self.observation = self.make_observaton(obs)

        return self.observation

    def automatic_throttle_control(self, target_speed, curr_state, accel, steer):
        if curr_state["speedX"] < target_speed - (steer*50):
            accel += .01
        else:
            accel -= .01

        if accel > 0.2:
            accel = 0.2

        if curr_state["speedX"] < 10:
            accel += 1 / (curr_state["speedX"] + .1)

        # Traction Control System
        if ((curr_state["wheelSpinVel"][2]+curr_state["wheelSpinVel"][3]) -
           (curr_state["wheelSpinVel"][0]+curr_state["wheelSpinVel"][1]) > 5):
            accel -= .2

        return accel

    def automatic_gearbox(self, speed, gear):
        #  Automatic Gear Change by Snakeoil is possible
        if speed > 50:
            gear = 2
        if speed > 80:
            gear = 3
        if speed > 110:
            gear = 4
        if speed > 140:
            gear = 5
        if speed > 170:
            gear = 6

        return gear

    def agent_to_torcs(self, u):
        torcs_action = {"steer": u[0]}

        if self.throttle is True:
            # composite throttle/brake, reduces search space size
            if(u[1] > 0):
                # accelerator is upper half
                torcs_action.update({"accel": u[1]})
                torcs_action.update({"brake": 0})
            else:
                # brake is lower half
                torcs_action.update({"accel": 0})
                torcs_action.update({"brake": u[1]})


        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({"gear": int(u[2])})

        return torcs_action

    def make_observaton(self, raw_obs):
        """
        returns a numpy array with the normalized state values specified in state_filter
        """
        obs = []
        for cat in self.state_filter:
            par = np.array(raw_obs[cat], dtype=np.float32)/self.state_filter[cat]
            obs.append(par)
        print(obs)

        return obs
