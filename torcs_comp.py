from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import subprocess
import time

from rewards import custom_reward


class TorcsEnv:
    terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50
    boring_speed = 1

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False, state_filter = None):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.initial_run = True
        self.img_width, self.img_height = 64, 64
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
        self.reset_torcs()

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
        shape = 0
        if "angle" in self.state_filter:
            shape += 1
            high = np.append(high, 1.0)
            low = np.append(low, -1.0)
        if "track" in self.state_filter:
            # the track rangefinder is made of 19 separate values
            shape += 19
            for i in range(19):
                high = np.append(high, 1.0)
                low = np.append(low, 0.0)
        if "trackPos" in self.state_filter:
            shape += 1
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "speedX" in self.state_filter:
            shape += 1
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "speedY" in self.state_filter:
            shape += 1
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "speedZ" in self.state_filter:
            shape += 1
            high = np.append(high, np.inf)
            low = np.append(low, -np.inf)
        if "wheelSpinVel" in self.state_filter:
            # one value each wheel
            shape += 4
            for i in range(4):
                high = np.append(high, np.inf)
                low = np.append(low, 0.0)
        if "rpm" in self.state_filter:
            shape += 1
            high = np.append(high, np.inf)
            low = np.append(low, 0.0)

        if self.vision:
            shape += self.img_width * self.img_height
            for i in range(self.img_width * self.img_height):
                high = np.append(high, 255)
                low = np.append(low, 0.0)

        self.observation_space = spaces.Box(low=low, high=high, shape=(shape,))

    def step(self, u):
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_prev = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        reward = custom_reward(obs, obs_prev)
        # Termination judgement #########################
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        speed = np.array(obs['speedX'])
        episode_terminate = False
        # if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
        #    reward = -200
        #    episode_terminate = True
        #    client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step:
            # Episode terminates if the agent is too slow
            if speed < self.boring_speed:
               episode_terminate = True
               client.R.d['meta'] = True
            # Episode terminates if the progress of agent is small
            progress = reward
            if progress < self.termination_limit_progress:
               episode_terminate = True
               client.R.d['meta'] = True



        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta']

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        subprocess.Popen(["pkill", "torcs"], stdout=subprocess.DEVNULL)

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        subprocess.Popen(["pkill", "torcs"], stdout=subprocess.DEVNULL)
        time.sleep(0.5)
        if self.vision is True:
            subprocess.Popen(["torcs", "-nofuel", "-nodamage", "-nolaptime", "-vision"], stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL)
        else:
            subprocess.Popen(["torcs", "-nofuel", "-nodamage", "-nolaptime"], stderr=subprocess.STDOUT,  stdout=subprocess.DEVNULL)
        time.sleep(0.5)
        subprocess.Popen(["sh", "autostart.sh"], stdout=subprocess.DEVNULL)
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            # composite throttle/brake, reduces search space size
            if(u[1] > 0):
                # accelerator is upper half
                torcs_action.update({'accel': u[1]})
                torcs_action.update({'brake': 0})
            else:
                # brake is lower half
                torcs_action.update({'accel': 0})
                torcs_action.update({'brake': u[1]})


        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[2])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)

        return np.array([r, g, b], dtype=np.uint8)

    def make_observaton(self, raw_obs):
        """
        returns a numpy array with the normalized state values specified in state_filter
        """
        obs = np.array([])
        for cat in self.state_filter:
            par = np.array(raw_obs[cat], dtype=np.float32)/self.state_filter[cat]
            obs = np.append(obs, par)
        if self.vision:
            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs["img"])
            obs = np.append(obs, image_rgb)

        return obs
