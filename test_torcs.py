from torcs_comp import TorcsEnv
import os
import numpy as np

from agent_ddpg.ddpg import DDPG

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


N_EPISODES = 1000

def main(vision = True):
    # dict with key: observation name, val: scale
    state_filter = {}
    # state_filter["track"] = 200.0
    # state_filter["speedX"] = 300.0
    # state_filter["speedY"] = 300.0
    # state_filter["speedZ"] = 300.0
    # state_filter["wheelSpinVel"] = 1.0
    # state_filter["rpm"] = 10000

    # Instantiate the environment
    env = TorcsEnv(throttle = False, vision = vision, state_filter = state_filter)
    action_dims = [env.action_space.shape[0]]
    state_dims = [env.observation_space.shape[0]]  # sensors input
    action_boundaries = [env.action_space.low[0], env.action_space.high[0]]

    # agent = DDPG(state_dims = state_dims, action_dims = action_dims,
    #     action_boundaries = action_boundaries, actor_lr = 1e-6,
    #     critic_lr = 2*1e-6, batch_size = 32, gamma = 0.99, rand_steps = 0,
    #     buf_size = int(1e2), tau = 0.001, fcl1_size = 128, fcl2_size = 64)

    np.random.seed(0)
    scores = []
    for i in range(N_EPISODES):
        state = env.reset()
        terminal = False
        score = 0

        while not terminal:
            #predict new action
            action = np.random.random(action_dims) # agent.get_action(state, i)
            #perform the transition according to the predicted action
            state_new, reward, terminal = env.step(action)

            #store the transaction in the memory
            # agent.remember(state, state_new, action, reward, terminal)
            #adjust the weights according to the new transaction
            # agent.learn(i)
            #iterate to the next state
            state = state_new
            score += reward
            scores.append(score)
        print(bcolors.OKBLUE + "Iteration {:d} --> score {:.2f}. Average score {:.2f}".format(i, score, np.mean(scores)) + bcolors.ENDC)


if __name__ == "__main__":
    #tell tensorflow to train with GPU 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(vision = True)
