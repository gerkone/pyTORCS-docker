import numpy as np
import importlib.util

from torcs_client.torcs_comp import TorcsEnv
from torcs_client.utils import bcolors

N_EPISODES = 1000

def agent_from_module(mod_name, run_path):
    spec = importlib.util.spec_from_file_location(mod_name, run_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, mod_name)

def main(verbose = False, hyperparams = None, sensors = None, image_name = "gerkone/torcs",
    environment = None, algo_name = None, algo_path = None, img_width = 640, img_height = 480):
    # Instantiate the environment
    env = TorcsEnv(throttle = environment["throttle"], gear_change = environment["gear_change"], verbose = verbose, state_filter = sensors,
        image_name = image_name, img_width = img_width, img_height = img_height)
    action_dims = [env.action_space.shape[0]]
    state_dims = [env.observation_space.shape[0]]  # sensors input
    action_boundaries = [env.action_space.low[0], env.action_space.high[0]]

    agent_class = agent_from_module(algo_name, algo_path)

    agent = agent_class(state_dims = state_dims, action_dims = action_dims,
                action_boundaries = action_boundaries, hyperparams = hyperparams)

    np.random.seed(0)
    scores = []
    for i in range(N_EPISODES):
        state = env.reset()
        terminal = False
        score = 0

        while not terminal:
            #predict new action
            action = agent.get_action(state, i)
            #perform the transition according to the predicted action
            state_new, reward, terminal = env.step(action)

            #store the transaction in the memory
            if hasattr(agent, 'remember'):
                if callable(agent.remember):
                    agent.remember(state, state_new, action, reward, terminal)
            #adjust the weights according to the new transaction
            if hasattr(agent, 'learn'):
                if callable(agent.learn):
                    agent.learn(i)
            #iterate to the next state
            state = state_new
            score += reward
        scores.append(score)
        print(bcolors.OKBLUE + "Iteration {:d} --> score {:.2f}. Running average {:.2f}".format(i, score, np.mean(scores)) + bcolors.ENDC)
