import numpy as np

class ReplayBuffer(object):
    """
    Buffer for experience replay
    """
    def __init__(self, buf_size, input_shape, output_shape):
        self.buf_size = buf_size
        self._buf_ctr = 0

        self.state_buf = np.zeros((self.buf_size, *input_shape))
        self.trans_buf = np.zeros((self.buf_size, *input_shape))
        self.action_buf = np.zeros((self.buf_size, *output_shape))
        self.reward_buf = np.zeros((self.buf_size, 1))
        self.terminal_buf = np.zeros(self.buf_size, dtype=np.float32)


    def remember(self, state, state_new, action, reward, terminal):
        state = np.hstack(list(state.values()))
        state_new = np.hstack(list(state_new.values()))

        i = self._buf_ctr % self.buf_size
        self.state_buf[i] = state
        self.trans_buf[i] = state_new
        self.action_buf[i] = action
        self.reward_buf[i] = reward
        self.terminal_buf[i] = terminal
        self._buf_ctr+=1;

    def sample(self, batch_size):
        """
        Batched sample from the experience buffer
        """
        i_max = min(self.buf_size, self._buf_ctr)
        batch = np.random.choice(i_max, batch_size)

        states = self.state_buf[batch]
        actions = self.action_buf[batch]
        rewards = self.reward_buf[batch]
        states_n = self.trans_buf[batch]
        terminal = self.terminal_buf[batch]

        return (states, actions, rewards, terminal, states_n)

    def isReady(self, batch_size):
        """
        Buffer is ready to be sampled from
        """
        return(self._buf_ctr > batch_size)
