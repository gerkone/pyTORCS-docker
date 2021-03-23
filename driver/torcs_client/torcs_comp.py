from gym import spaces
import numpy as np
# from os import path
import numpy as np
import copy
import collections as col
import matplotlib.pyplot as plt


import torcs_client.snakeoil3_gym as snakeoil3
from torcs_client.reward import custom_reward
from torcs_client.terminator import custom_terminal
from torcs_client.utils import start_container, reset_torcs

class TorcsEnv:
    terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50
    boring_speed = 1

    initial_reset = True

    def __init__(self, vision = False, throttle = False, gear_change = False, state_filter = None, img_width = 0, img_height = 0, verbose = False, image_name = "gerkone/vtorcs"):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.verbose = verbose

        self.image_name = image_name

        if self.image_name != "0":
            # start torcs container
            self.container_id = start_container(self.image_name, self.verbose)
        else:
            self.container_id = "0"

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
            self.state_filter = state_filter
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

        # run torcs and start practice run
        reset_torcs(self.container_id, self.vision)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        high = np.array([])
        low = np.array([])
        if "angle" in self.state_filter:
            high = np.append(high, 1.0)
            low = np.append(low, -1.0)
        if "track" in self.state_filter:
            # the track rangefinder is made of 19 separate values
            for i in range(19):
                high = np.append(high, 1.0)
                low = np.append(low, 0.0)
        if "trackPos" in self.state_filter:
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "speedX" in self.state_filter:
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "speedY" in self.state_filter:
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "speedZ" in self.state_filter:
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "wheelSpinVel" in self.state_filter:
            # one value each wheel
            for i in range(4):
                high = np.append(high, np.inf)
                low = np.append(low, 0.0)
        if "rpm" in self.state_filter:
            high = np.append(high, np.inf)
            low = np.append(low, 0.0)

        self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):

        # convert thisAction to the actual torcs actionstr
        this_action = self.agent_to_torcs(u)

        # current observation
        curr_state = self.client.S.d

        # Apply Action
        action_torcs = self.client.R.d

        # Steering
        action_torcs["steer"] = this_action["steer"]  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if curr_state["speedX"] < target_speed - (self.client.R.d["steer"]*50):
                self.client.R.d["accel"] += .01
            else:
                self.client.R.d["accel"] -= .01

            if self.client.R.d["accel"] > 0.2:
                self.client.R.d["accel"] = 0.2

            if curr_state["speedX"] < 10:
                self.client.R.d["accel"] += 1/(curr_state["speedX"]+.1)

            # Traction Control System
            if ((curr_state["wheelSpinVel"][2]+curr_state["wheelSpinVel"][3]) -
               (curr_state["wheelSpinVel"][0]+curr_state["wheelSpinVel"][1]) > 5):
                action_torcs["accel"] -= .2
        else:
            action_torcs["accel"] = this_action["accel"]
            action_torcs["brake"] = this_action["brake"]

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs["gear"] = this_action["gear"]
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs["gear"] = 1
            if self.throttle:
                if curr_state["speedX"] > 50:
                    action_torcs["gear"] = 2
                if curr_state["speedX"] > 80:
                    action_torcs["gear"] = 3
                if curr_state["speedX"] > 110:
                    action_torcs["gear"] = 4
                if curr_state["speedX"] > 140:
                    action_torcs["gear"] = 5
                if curr_state["speedX"] > 170:
                    action_torcs["gear"] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_prev = copy.deepcopy(curr_state)

        # One-Step Dynamics Update #################################
        # Apply the Agent"s action into torcs
        self.client.respond_to_server()
        # Get the response of TORCS
        self.client.get_servers_input()

        # Get the current full-observation from torcs
        obs = curr_state

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        reward = custom_reward(obs, obs_prev)
        # Termination judgement #########################
        episode_terminate = custom_terminal(obs, reward, terminal_judge_start = self.terminal_judge_start, time_step = self.time_step,
                            termination_limit_progress = self.termination_limit_progress, boring_speed = self.boring_speed)
        self.client.R.d["meta"] = episode_terminate

        if episode_terminate: # Send a reset signal
            self.initial_run = False
            self.client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, episode_terminate

    def reset(self, relaunch=False):
        if self.verbose: print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                reset_torcs(self.container_id, self.vision)
                if self.verbose: print("### TORCS is RELAUNCHED ###")

        if self.initial_reset:
            # create new torcs client if first reset
            self.client = snakeoil3.Client(p=3101, vision=self.vision, verbose = self.verbose, image_name = self.image_name)
        else:
            # Open new UDP to vtorcs
            self.client.reset()

        self.client.MAX_STEPS = np.inf

        self.client.get_servers_input()  # Get the initial input from torcs

        obs = self.client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        subprocess.Popen(["pkill", "torcs"], stdout=subprocess.DEVNULL)

    def get_obs(self):
        return self.observation

    def agent_to_torcs(self, u):
        torcs_action = {"steer": u[0]}

        if self.throttle is True:  # throttle action is enabled
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


    def raw_vision_to_rgb(self, obs_image_vec):
        """
        Transforms the raw byte array from vision in 3 two dimensional matrices
        """
        image_vec =  obs_image_vec

        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (self.img_width, self.img_height)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)

        r = np.flip(r, axis = 0)
        g = np.flip(g, axis = 0)
        b = np.flip(b, axis = 0)

        # if not hasattr(self, "plt"):
        #     self.fig, ax = plt.subplots(1,1)
        #     image = np.array(r)
        #     self.im = ax.imshow(image)
        #
        # self.im.set_data(np.array(r))
        # self.fig.canvas.draw_idle()
        # plt.pause(1)

        return np.array([r, g, b], dtype=np.uint8)

    def make_observaton(self, raw_obs):
        """
        returns a numpy array with the normalized state values specified in state_filter
        """
        obs = []
        for cat in self.state_filter:
            par = np.array(raw_obs[cat], dtype=np.float32)/self.state_filter[cat]
            obs.append(par)
        if self.vision:
            # Get RGB from observation
            image_rgb = self.raw_vision_to_rgb(raw_obs["img"])
            obs.append(image_rgb)

        return obs
