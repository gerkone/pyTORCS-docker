from gym import spaces
import numpy as np
# from os import path
import copy
import collections as col
import matplotlib.pyplot as plt
import sys, signal

from torcs_client.torcs_client import Client
from torcs_client.reward import TimeReward, LocalReward
from torcs_client.terminator import custom_terminal
from torcs_client.utils import SimpleLogger as log, start_container, reset_torcs, kill_torcs, kill_container, change_track, change_car, change_driver

class TorcsEnv:
    def __init__(self, throttle = False, gear_change = False, car = "car1-trb1",  state_filter = None, target_speed = 50, sid = "SCR", port = 3001,
                driver_id = "0", driver_module = "scr_server", img_width = 640, img_height = 480, verbose = False, image_name = "gerkone/torcs"):

        self.throttle = throttle
        self.gear_change = gear_change
        self.target_speed = target_speed

        self.verbose = verbose

        self.image_name = image_name

        self.sid = sid

        self.port = port

        if self.image_name != "0":
            # start torcs container
            self.container_id = start_container(self.image_name, self.verbose, self.port)
        else:
            self.container_id = "0"

        self.shift_debounce = 0

        self.img_width = img_width
        self.img_height = img_height


        # reward class
        # TODO parmametric change
        # locrw = LocalReward()
        # reward = LocalReward.local_reward(obs, obs_prev, action, action_prev, cur_step, terminal)
        self.rewarder = TimeReward()

        # TODO support other races
        self.race_type = "practice"
        self.tracks_categories = {}
        self.tracks_categories["dirt"] = ["dirt-1", "dirt-2", "dirt-3", "dirt-4", "dirt-5", "dirt-6", "mixed-1", "mixed-2"]
        self.tracks_categories["road"] = ["alpine-1", "corkscrew", "e-track-3", "g-track-2", "ole-road-1", "street-1", "alpine-2",
                        "e-track-6", "g-track-3", "ruudskogen", "wheel-1", "brondehach", "e-track-2", "forza", "spring", "wheel-2",
                        "aalborg", "e-track-1", "e-track-5", "e-track-1", "e-track-5", "eroad", "e-track-4", "g-track-1"]
        self.tracks_categories["oval"] = ["a-speedway", "b-speedway", "e-speedway", "g-speedway", "michigan", "c-speedway", "d-speedway", "f-speedway"]

        # restart request ( relaunches torcs on environment reset )
        self.restart_needed = True

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

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

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

        self.observation_space = spaces.Box(low = np.float32(low), high = np.float32(high), dtype = np.float32)

        change_car(self.race_type, car)
        if self.verbose: log.info("Car: {}".format(car))

        change_driver(self.race_type, driver_id, driver_module)
        if self.verbose: log.info("Now driving: {} {}".format(driver_module, driver_id))

        # kill torcs on sigint, avoid leaving the open window
        def kill_torcs_and_close(sig, frame):
            kill_torcs(self.container_id)
            sys.exit(0)

        signal.signal(signal.SIGINT, kill_torcs_and_close)

    def step(self, u):
        # get the state from torcs - simulation step
        error_restart = self.client.get_servers_input()

        # convert u to the actual torcs actionstr
        action = self.agent_to_torcs(u)

        # current observation
        curr_state = self.client.S.d

        self.client.R.d["steer"] = action["steer"]

        if self.throttle is False:
            try:
                self.client.R.d["accel"] = self.automatic_throttle_control(self.target_speed, curr_state, self.client.R.d["accel"], self.client.R.d["steer"])
            except Exception:
                self.client.R.d["accel"] = 0
        else:
            self.client.R.d["accel"] = action["accel"]
            self.client.R.d["brake"] = action["brake"]

        try:
            if self.gear_change is False:
                # debounce used to avoid shifting 2 or more gears at once ( engine did not have the time to slow down )
                self.shift_debounce -=  1
                self.client.R.d["gear"] = self.automatic_gearbox(curr_state["rpm"], self.client.R.d["gear"])
            else:
                self.client.R.d["gear"] = action["gear"]
        except Exception:
            self.client.R.d["gear"] = 0

        # Apply the agent"s action into torcs
        self.client.respond_to_server()

        # initialize previous observation (for reward and termination)
        if self.curr_step == 0:
            self.obs_prev = copy.deepcopy(curr_state)
            self.action_prev = copy.deepcopy(action)

        ################### Termination ###################
        try:
            episode_terminate = custom_terminal(curr_state, curr_step = self.curr_step)
        except Exception:
            episode_terminate = False

        ################### Reward ###################
        try:
            reward = self.rewarder.get_reward(curr_state, self.obs_prev, action, self.action_prev, self.curr_step, terminal = episode_terminate, track = self.track)
        except Exception:
            reward = 0

        if episode_terminate:
            if self.verbose: log.info("Episode terminated by condition")

        if error_restart:
            if self.verbose: log.alert("Episode terminated by error timeout")
            episode_terminate = True

        self.curr_step += 1

        self.obs_prev = copy.deepcopy(curr_state)

        return self.make_observaton(curr_state), reward, episode_terminate

    def reset(self):
        """
        episode either terminated or just started
        """

        if self.verbose: log.info("Reset torcs")

        vision = "img" in self.state_filter
        first_run = not hasattr(self, "client")

        if self.restart_needed:
            self.restart_needed = False
            # launch torcs for the first time
            reset_torcs(self.container_id, vision, True)
        if first_run:
            # create new torcs client - after first torcs launch
            self.client = Client(port = self.port, verbose = self.verbose, sid = self.sid,
                container_id = self.container_id, vision = vision, img_width = 640, img_height = 480)
        else:
            # restart torcs without closing - tell scr_server to restart race
            self.client.R.d["meta"] = True
            self.client.respond_to_server()
            # reset UDP, reset client
            self.client.restart()

        self.curr_step = 0

        # Get the initial full-observation from torcs
        obs = self.client.S.d

        # reset reward params
        self.rewarder.reset()

        return self.make_observaton(obs)

    def terminate(self):
        kill_torcs(self.container_id)
        kill_container(self.container_id)
        self.client.shutdown()

    def set_track(self, track):
        if track is None:
            track = "g-track-1"
        change_track(self.race_type, track, self.tracks_categories)
        # torcs needs to be restarted for the config to be loaded
        self.restart_needed = True
        self.track = track

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

    def automatic_gearbox(self, rpm, gear):
        if rpm > 9500 and gear < 6 and self.shift_debounce <= 0:
            self.shift_debounce = 5
            gear += 1
        if rpm < 4500 and gear > 1  and self.shift_debounce <= 0:
            self.shift_debounce = 5
            gear -= 1

        return gear

    def get_max_packets(self):
        return self.client.max_packets

    def agent_to_torcs(self, u):
        torcs_action = {"steer": u[0]}

        if self.throttle is True:
            # composite throttle/brake, reduces search space size
            if(u[1] > 0):
                # accelerator is upper half
                torcs_action.update({"accel": u[1]})
                torcs_action.update({"brake": 0})
            else:
                # brake is inveerted lower half
                torcs_action.update({"accel": 0})
                torcs_action.update({"brake": -u[1]})


        if self.gear_change is True:
            torcs_action.update({"gear": int(u[2])})

        return torcs_action

    def make_observaton(self, raw_obs):
        """
        returns a numpy array with the normalized state values specified in state_filter
        """
        obs = {}
        for cat in self.state_filter:
            try:
                par = np.array(raw_obs[cat], dtype=np.float32)/self.state_filter[cat]
                obs[cat] = par
            except Exception:
                # one of your sensors was not in the incoming string
                pass

        return obs
