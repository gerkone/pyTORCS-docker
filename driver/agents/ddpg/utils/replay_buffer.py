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

        self.state_buf = np.zeros((self.buf_size, *input_shape))
        self.trans_buf = np.zeros((self.buf_size, *input_shape))
        self.action_buf = np.zeros((self.buf_size, *output_shape))
        self.reward_buf = np.zeros((self.buf_size, 1))
        self.terminal_buf = np.zeros(self.buf_size, dtype=np.float32)

        self.input_shape = input_shape[0]
        self.output_shape = output_shape[0]

    def unpack_state(self, state):
        """
        state dict to state array if fixed order
        """
        state_array = np.zeros(self.input_shape)

        state_array[0] = state["speedX"]
        state_array[1] = state["speedY"]
        state_array[2] = state["speedZ"]
        state_array[3] = state["angle"]
        state_array[4] = state["trackPos"]
        state_array[5:9] = state["wheelSpinVel"]
        state_array[9:28] = state["track"]

        return state_array


    def remember(self, state, state_new, action, reward, terminal):
        try:
            state = self.unpack_state(state)
            state_new = self.unpack_state(state_new)
        except:
            # state is already in ndarray form
            pass

        i = self._buf_ctr % self.buf_size
        self.state_buf[i] = state
        self.trans_buf[i] = state_new
        self.action_buf[i] = action
        self.reward_buf[i] = reward
        self.terminal_buf[i] = terminal
        self._buf_ctr += 1


    def sample(self, batch_size, trajectory_mode = False):
        """
        Batched sample from the experience buffer
        """

        if trajectory_mode == False:
            i_max = min(self.buf_size, self._buf_ctr)
            batch = np.random.choice(i_max, batch_size)
        else:
            # sequential samples
            batch = np.arange(self._ret_ctr, self._ret_ctr + batch_size, 1)

        states = self.state_buf[batch]
        actions = self.action_buf[batch]
        rewards = self.reward_buf[batch]
        states_n = self.trans_buf[batch]
        terminal = self.terminal_buf[batch]

        if trajectory_mode == True:
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

    def isFull(self):
        """
        Buffer is full, new data will wrap around
        """
        return(self._buf_ctr > self.buf_size)

    def __len__(self):
        return min(self._buf_ctr, self.buf_size)

    def reset(self):
        self._buf_ctr = 0
        self._ret_ctr = 0
        self.done = False

        self.state_buf = np.zeros((self.buf_size, self.input_shape))
        self.trans_buf = np.zeros((self.buf_size, self.input_shape))
        self.action_buf = np.zeros((self.buf_size, self.output_shape))
        self.reward_buf = np.zeros((self.buf_size, 1))
        self.terminal_buf = np.zeros(self.buf_size, dtype=np.float32)
