import numpy as np
# from agents.ddpg.ddpg import DDPG
from torcs_client.torcs_comp import TorcsEnv


N_EPISODES = 1000

def main(verbose = False, hyperparams = None, sensors = None, image_name = "gerkone/vtorcs", img_width = 64, img_height = 64):
    # Instantiate the environment
    env = TorcsEnv(throttle = False, vision = True, verbose = verbose, state_filter = sensors,
            image_name = image_name, img_width = img_width, img_height = img_height)
    action_dims = [env.action_space.shape[0]]
    state_dims = [env.observation_space.shape[0]]  # sensors input
    action_boundaries = [env.action_space.low[0], env.action_space.high[0]]

    # agent = DDPG(state_dims = state_dims, action_dims = action_dims, action_boundaries = action_boundaries,
    #     actor_lr = hyperparams["actor_lr"], critic_lr = hyperparams["critic_lr"], batch_size = hyperparams["batch_size"],
    #     gamma = hyperparams["gamma"], rand_steps = hyperparams["rand_steps"], buf_size = int(hyperparams["buf_size"]),
    #     tau = hyperparams["tau"], fcl1_size = hyperparams["fcl1_size"], fcl2_size = hyperparams["fcl2_size"])


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
