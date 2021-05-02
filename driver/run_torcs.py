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
    train_req = training["train_req"]

    # Instantiate the environment
    env = TorcsEnv(throttle = training["throttle"], gear_change = training["gear_change"], verbose = verbose, state_filter = sensors,
            target_speed = training["target_speed"], max_steps = max_steps, image_name = image_name, img_width = img_width, img_height = img_height)

    action_dims = [env.action_space.shape[0]]
    state_dims = [env.observation_space.shape[0]]  # sensors input
    action_boundaries = [env.action_space.low[0], env.action_space.high[0]]

    vision = "img" in sensors
    # stacked frames for image input
    use_stacked_frames = vision and stack_depth > 1


    agent_class = agent_from_module(algo_name, algo_path)

    os.path.isdir('new_folder')

    agent = agent_class(state_dims = state_dims, action_dims = action_dims,
            action_boundaries = action_boundaries, hyperparams = hyperparams)

    _, columns = os.popen('stty size', 'r').read().split()

    scores = []
    curr_step = 0

    if use_stacked_frames:
        frame_stack = collections.deque(maxlen=stack_depth)

    log.separator(int(columns) / 2)

    collected_steps = 0

    # buffer episodes in between training steps
    episode_buffer = np.empty(max(max_steps, train_req) * 2, dtype = object)

    for i in range(episodes):
        state = env.reset()
        time_start = time.time()
        terminal = False
        score = 0
        avg_loss = []
        curr_step = 0
        if vision:
            state["img"] = resize_frame(state["img"], img_width, img_height)
        if use_stacked_frames:
            frame_stack.clear()
            frame_stack.append(state["img"])
            frame_stack.append(state["img"])
            frame_stack.append(state["img"])
            state["img"] = frame_stack

        log.info("Episode {}/{} started".format(i, episodes))

        while not terminal and curr_step < max_steps:
            # time_1 = time.time()
            if curr_step >= max_steps:
                if self.verbose: log.info("Episode terminated by steps: {} steps done.".format(max_steps))
            # predict new action
            action = agent.get_action(state, i)
            # perform the transition according to the choosen action
            state_new, reward, terminal = env.step(action)
            if vision:
                state_new["img"] = resize_frame(state_new["img"], img_width, img_height)
            if use_stacked_frames:
                frame_stack.append(state_new["img"])
                state_new["img"] = frame_stack

            # save step in buffer
            episode_buffer[collected_steps] = (state, state_new, action, reward, terminal)
            collected_steps += 1
            #iterate to the next
            state = state_new
            curr_step += 1
            score += reward
        time_end = time.time()

        scores.append(score)
        duration = 1000.0 * (time_end - time_start)
        avg_iteration = duration / curr_step
        packet_loss = (avg_iteration / (1000 / 50)) * 100
        log.info("Iteration {:d} --> Duration {:.2f} ms. Score {:.2f}. Running average {:.2f}".format(
            i, duration, score, np.mean(scores)))
        if packet_loss > 350:
            if verbose: log.alert("High packet loss: {:.2f}%. Running {:.2f} ms behind torcs.".format(packet_loss, (avg_iteration - 1000/50) * env.get_max_packets()))


        # accumulate some training data before training
        if collected_steps >= train_req:

            has_remember = hasattr(agent, "remember") and callable(agent.remember)
            if has_remember:
                i = 0
                for (state, state_new, action, reward, terminal) in episode_buffer[0:collected_steps - 1]:
                    i += 1
                    # store the transaction in the memory
                    agent.remember(state, state_new, action, reward, terminal)

            ##################### TRAINING #####################
            has_train = hasattr(agent, "learn") and callable(agent.learn)
            if has_train:
                log.info("Starting training: {:d} epochs over {:d} collected steps".format(n_epochs, collected_steps))
                time_start = time.time()
                for e in range(n_epochs):
                    for i in range(collected_steps):
                        # adjust the weights according to the new transaction
                        loss = agent.learn(i)
                        avg_loss.append(loss)
                    if verbose: log.training("Epoch {}. ".format(e + 1), loss)
                time_end = time.time()
                log.info("Completed {:d} epochs. Duration {:.2f} ms. Average loss {:.3f}".format(
                    n_epochs, 1000.0 * (time_end - time_start), np.mean(avg_loss)))
                # reset lived collection steps
                collected_steps = 0
                # empty episode buffer
                del episode_buffer
                episode_buffer = np.empty(max(max_steps, train_req) * 2, dtype = object)

            has_save = hasattr(agent, "save_models") and callable(agent.save_models)
            if has_save:
                log.info("Saving models...")
                agent.save_models()
            log.separator(int(columns) / 2)

    log.info("All done. Closing...")
    env.terminate()
    input("...")
