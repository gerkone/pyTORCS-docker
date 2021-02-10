from torcs_comp import TorcsEnv
import os
import numpy as np

from agent.ddpg import Agent


N_EPISODES = 1000


def state_filter(state):
    return np.hstack((state.angle,
        state.track,
        state.trackPos,
        state.speedX,
        state.speedY,
        state.speedZ,
        state.wheelSpinVel/100.0,
        state.rpm))

def main():
    # Instantiate the environment
    # env = gym.make("Torcs-v0", vision=False, rendering=False, throttle=True, gear_change=False)
    env = TorcsEnv(throttle=False)
    action_dims = [env.action_space.shape[0]] # steering/acceleration/brake
    state_dims = [29]  # sensors input
    action_boundaries = [-1,1]
    agent = Agent(state_dims = state_dims, action_dims = action_dims,
    action_boundaries = action_boundaries, actor_lr = 1e-6,
    critic_lr = 2*1e-6, batch_size = 32, gamma = 0.99, rand_steps = 0,
    buf_size = int(1e5), tau = 0.001, fcl1_size = 128, fcl2_size = 64)

    np.random.seed(0)
    scores = []
    for i in range(N_EPISODES):
        state = env.reset()
        state = state_filter(state)
        terminal = False
        score = 0

        while not terminal:
            #predict new action
            action = agent.get_action(state, i)
            #perform the transition according to the predicted action
            state_new, reward, terminal, _ = env.step(action)
            state_new = state_filter(state_new)
            #store the transaction in the memory
            agent.remember(state, state_new, action, reward, terminal)
            #adjust the weights according to the new transaction
            agent.learn(i)
            #iterate to the next state
            state = state_new
            score += reward
            scores.append(score)
        print("Iteration {:d} --> score {:.2f}. Average score {:.2f}".format(i, score, np.mean(scores)))


if __name__ == "__main__":
    #tell tensorflow to train with GPU 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
