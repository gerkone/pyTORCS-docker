import numpy as np
import collections
import time
import os

from torcs_client.torcs_comp import TorcsEnv
from torcs_client.utils import SimpleLogger as log, resize_frame, agent_from_module

def main(verbose = False, hyperparams = None, sensors = None, image_name = "gerkone/torcs", driver = None,
        privileged = False, training = None, algo_name = None, algo_path = None, stack_depth = 1, img_width = 640, img_height = 480):

    max_steps = 1000
    n_epochs = 5
    episodes = 1000
    train_req = 1000

    if "max_steps" in training.keys(): max_steps = training["max_steps"]
    if "epochs" in training.keys(): n_epochs = training["epochs"]
    if "episodes" in training.keys(): episodes = training["episodes"]
    if "train_req" in training.keys(): train_req = training["train_req"]

    track_list = [None]
    car = None

    # never stop due to steps
    infinite = max_steps == -1

    if "track" in training.keys(): track_list = training["track"]
    if "car" in training.keys(): car = training["car"]

    if driver != None:
        sid = driver["sid"]
        ports = driver["ports"]
        driver_id = driver["index"]
        driver_module = driver["module"]
    else:
        sid = "SCR"
        port = [3001]
        driver_id = "0"
        driver_module = "scr_server"

    # Instantiate the environment
    env = TorcsEnv(throttle = training["throttle"], gear_change = training["gear_change"], car = car,
            verbose = verbose, state_filter = sensors, target_speed = training["target_speed"], sid = sid,
            ports = ports, driver_id = driver_id, driver_module = driver_module, image_name = image_name,
            privileged = privileged, img_width = img_width, img_height = img_height)

    action_dims = [env.action_space.shape[0]]
    state_dims = [env.observation_space.shape[0]]  # sensors input
    action_boundaries = [env.action_space.low[0], env.action_space.high[0]]

    vision = "img" in sensors
    # stacked frames for image input
    use_stacked_frames = vision and stack_depth > 1

    agent_class = agent_from_module(algo_name, algo_path)

    agent = agent_class(state_dims = state_dims, action_dims = action_dims,
            action_boundaries = action_boundaries, hyperparams = hyperparams)

    _, columns = os.popen("stty size", "r").read().split()

    scores = []
    curr_step = 0

    if use_stacked_frames:
        frame_stack = collections.deque(maxlen=stack_depth)


    log.separator(int(columns) / 2)

    collected_steps = 0

    if not infinite:
        # buffer episodes in between training steps
        episode_buffer = np.empty(max(max_steps, train_req) * 2, dtype = object)

    for track in track_list:
        log.info("Starting {} episodes on track {}".format(episodes, track))

        env.set_track(track)
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
                for k in range(stack_depth):
                    frame_stack.append(state["img"])
                state["img"] = frame_stack

            log.info("Episode {}/{} started".format(i + 1, episodes))

            while not terminal and (curr_step < max_steps or infinite):
                # time_1 = time.time()
                if not infinite:
                    if curr_step >= max_steps:
                        if self.verbose: log.info("Episode terminated by steps: {} steps done.".format(max_steps))
                # predict new action
                action = agent.get_action(state, i, track)
                # perform the transition according to the choosen action
                state_new, reward, terminal = env.step(action)
                if vision:
                    state_new["img"] = resize_frame(state_new["img"], img_width, img_height)
                if use_stacked_frames:
                    frame_stack.append(state_new["img"])
                    state_new["img"] = frame_stack

                if not infinite:
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
            if collected_steps >= train_req and not infinite:
                has_remember = hasattr(agent, "remember") and callable(agent.remember)
                if has_remember:
                    i = 0
                    for (state, state_new, action, reward, terminal) in episode_buffer[0:collected_steps]:
                        i += 1
                        # store the transaction in the memory
                        agent.remember(state, state_new, action, reward, terminal)

                ##################### TRAINING #####################
                has_train = hasattr(agent, "learn") and callable(agent.learn)
                if has_train:
                    log.info("Starting training: {:d} epochs over {:d} collected steps".format(n_epochs, collected_steps))
                    time_start = time.time()
                    for e in range(n_epochs):
                        # adjust the weights according to the new transaction
                        loss = agent.learn(i)
                        if loss != None:
                            avg_loss.append(loss)
                        if verbose: log.training("Epoch {}. ".format(e + 1), loss)
                    time_end = time.time()
                    log.info("Completed {:d} epochs. Duration {:.2f} ms. Average loss {:.3f}".format(
                        n_epochs, 1000.0 * (time_end - time_start), np.mean(avg_loss)))

                has_save = hasattr(agent, "save_models") and callable(agent.save_models)
                if has_save:
                    log.info("Saving models...")
                    agent.save_models()

                # reset lived collection steps
                collected_steps = 0
                # empty episode buffer
                del episode_buffer
                episode_buffer = np.empty(max(max_steps, train_req) * 2, dtype = object)

                log.separator(int(columns) / 2)

    log.info("All done. Closing...")
    env.terminate()
    if  hasattr(agent, "dataset_file"):
        # save dataset
        agent.dataset_file.close()
    input("...")
