import numpy as np

class ReplayBuffer(object):
    """
    Buffer for experience replay
    """
    def __init__(self, buf_size, input_shape, output_shape):
        self.buf_size = buf_size
        self._buf_ctr = 0
        self._ret_ctr = 0
        # buffer already emptied
        self.done = False

        self.input_shape = input_shape[0]
        self.output_shape = output_shape[0]

        self.state_buf = np.zeros((self.buf_size, self.input_shape))
        self.trans_buf = np.zeros((self.buf_size, self.input_shape))
        self.action_buf = np.zeros((self.buf_size, self.output_shape))
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
        self._buf_ctr += 1;

    def sample(self, batch_size):
        """
        Batched sample from the experience buffer
        """

        batch = np.arange(self._ret_ctr, self._ret_ctr + batch_size, 1)
        states = self.state_buf[batch]
        actions = self.action_buf[batch]
        rewards = self.reward_buf[batch]
        states_n = self.trans_buf[batch]
        terminal = self.terminal_buf[batch]

        self._ret_ctr = self._ret_ctr + batch_size
        if self._ret_ctr + batch_size > (self._buf_ctr % self.buf_size):
            self.done = True
            self._ret_ctr = 0
        return (states, actions, rewards, terminal, states_n)


    def isReady(self, batch_size):
        """
        Buffer is ready to be sampled from
        """
        return(self._buf_ctr > batch_size)

    def __len__(self):
        return min(self._buf_ctr, self.buf_size)

    def reset(self):
        del self.state_buf
        del self.trans_buf
        del self.action_buf
        del self.reward_buf
        del self.terminal_buf

        self._buf_ctr = 0
        self._ret_ctr = 0
        self.done = False

        self.state_buf = np.zeros((self.buf_size, self.input_shape))
        self.trans_buf = np.zeros((self.buf_size, self.input_shape))
        self.action_buf = np.zeros((self.buf_size, self.output_shape))
        self.reward_buf = np.zeros((self.buf_size, 1))
        self.terminal_buf = np.zeros(self.buf_size, dtype=np.float32)

    def get_trajectory(self):
        """
        return the whole trajectory of the registerd run
        """
        states = self.state_buf[0:self._buf_ctr]
        actions = self.action_buf[0:self._buf_ctr]
        return states, actions
