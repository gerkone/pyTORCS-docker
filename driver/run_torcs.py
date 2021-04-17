import numpy as np
import importlib.util
import collections
import time
import os

from torcs_client.torcs_comp import TorcsEnv
from torcs_client.utils import SimpleLogger as log, resize_frame

def agent_from_module(mod_name, run_path):
    spec = importlib.util.spec_from_file_location(mod_name, run_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, mod_name)

def main(verbose = False, hyperparams = None, sensors = None, image_name = "gerkone/torcs",
        training = None, algo_name = None, algo_path = None, stack_depth = 1, img_width = 640, img_height = 480):

    max_steps = training["max_steps"]
    n_epochs = training["epochs"]
    episodes = training["episodes"]

    # Instantiate the environment
    env = TorcsEnv(throttle = training["throttle"], gear_change = training["gear_change"], verbose = verbose, state_filter = sensors,
            max_steps = max_steps, image_name = image_name, img_width = img_width, img_height = img_height)

    action_dims = [env.action_space.shape[0]]
    state_dims = [env.observation_space.shape[0]]  # sensors input
    action_boundaries = [env.action_space.low[0], env.action_space.high[0]]

    # stacked frames for image input
    use_stacked_frames = "img" in sensors and stack_depth > 1


    agent_class = agent_from_module(algo_name, algo_path)

    agent = agent_class(state_dims = state_dims, action_dims = action_dims,
            action_boundaries = action_boundaries, hyperparams = hyperparams)

    _, columns = os.popen('stty size', 'r').read().split()

    np.random.seed(0)
    scores = []
    curr_step = 0

    if use_stacked_frames:
        frame_stack = collections.deque(maxlen=stack_depth)

    for i in range(episodes):
        state = env.reset()
        time_start = time.time()
        terminal = False
        score = 0
        avg_loss = []
        curr_step = 0
        if use_stacked_frames:
            frame_stack.clear()
            frame = resize_frame(state["img"], img_width, img_height)
            frame_stack.append(frame)
            frame_stack.append(frame)
            frame_stack.append(frame)
            state["img"] = frame_stack

        log.alert("Episode {}/{} started".format(i, episodes))
        log.separator(columns)

        while not terminal and curr_step < max_steps:
            # time_1 = time.time()
            if curr_step >= max_steps:
                if self.verbose: log.info("Episode terminated by steps: {} steps done.".format(max_steps))
            # predict new action
            action = agent.get_action(state, i)
            # perform the transition according to the choosen action
            state_new, reward, terminal = env.step(action)
            if use_stacked_frames:
                frame = resize_frame(state_new["img"], img_width, img_height)
                frame_stack.append(frame)
                state_new["img"] = frame_stack
            # store the transaction in the memory
            if hasattr(agent, "remember"):
                if callable(agent.remember):
                    agent.remember(state, state_new, action, reward, terminal)
            #iterate to the next
            state = state_new
            curr_step += 1
            score += reward
            # time_2 = time.time()
            # log.info("{:.2f}".format(1000.0 * (time_2 - time_1)))
        time_end = time.time()

        scores.append(score)
        log.info("Iteration {:d} --> Duration {:.2f} ms. Score {:.2f}. Running average {:.2f}".format(
            i, 1000.0 * (time_end - time_start), score, np.mean(scores)))
        log.separator(columns)
        if hasattr(agent, "learn"):
            if callable(agent.learn):
                log.alert("Starting training: {:d} epochs".format(n_epochs))
                time_start = time.time()
                for e in range(n_epochs):
                    # adjust the weights according to the new transaction
                    loss = agent.learn(i)
                    avg_loss.append(loss)
                    if verbose:
                        log.training("Epoch {}. ".format(e), loss)
                time_end = time.time()
                log.alert("Completed {:d} epochs. Duration {:.2f} ms. Average loss {:.3f}".format(
                    n_epochs, 1000.0 * (time_end - time_start), np.mean(avg_loss)))
