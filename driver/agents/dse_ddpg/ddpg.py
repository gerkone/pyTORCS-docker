import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

import cv2

from agents.ddpg.utils.replay_buffer import ReplayBuffer
from agents.ddpg.utils.action_noise import OUActionNoise
from agents.ddpg.network.actor import Actor
from agents.ddpg.network.critic import Critic

from torcs_client.utils import SimpleLogger as log

cur_dir = "driver/agents/dse_ddpg"
save_dir = cur_dir + "/model"
dse_dir = cur_dir + "/dse_model_plain"

class DDPG(object):
    """
    DDPG agent
    """
    def __init__(self, state_dims, action_dims, action_boundaries, hyperparams):

        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        actor_lr = hyperparams["actor_lr"]
        critic_lr = hyperparams["critic_lr"]
        batch_size = hyperparams["batch_size"]
        gamma = hyperparams["gamma"]
        buf_size = int(hyperparams["buf_size"])
        tau = hyperparams["tau"]
        fcl1_size = hyperparams["fcl1_size"]
        fcl2_size = hyperparams["fcl2_size"]
        self.guided_steps = hyperparams["guided_steps"] - 1

        # action size - needs to be hardcoded
        self.n_states = 28
        # state size
        self.n_actions = action_dims[0]
        self.batch_size = batch_size

        # environmental action boundaries
        self.lower_bound = action_boundaries[0]
        self.upper_bound = action_boundaries[1]

        self.img_height = hyperparams["img_height"]
        self.img_width = hyperparams["img_width"]
        self.stack_depth = hyperparams["stack_depth"]

        # experience replay buffer
        self._memory = ReplayBuffer(buf_size, input_shape = state_dims, output_shape = action_dims)
        # noise generator
        self._noise = OUActionNoise(mu=np.zeros(action_dims))
        # Bellman discount factor
        self.gamma = gamma

        self.prev_accel = 0

        # turn off most logging
        logging.getLogger("tensorflow").setLevel(logging.FATAL)

        # resnet estimator
        self.estimator_model = tf.keras.models.load_model(dse_dir)
        log.info("Loaded estimator model")

        # actor class
        self.actor = Actor(state_dims = state_dims, action_dims = action_dims,
                            lr = actor_lr, batch_size = batch_size, tau = tau,
                            upper_bound = self.upper_bound, save_dir = save_dir,
                            fcl1_size = fcl1_size, fcl2_size = fcl2_size)
        # critic class
        self.critic = Critic(state_dims = state_dims, action_dims = action_dims,
                            lr = critic_lr, batch_size = batch_size, tau = tau,
                            save_dir = save_dir, fcl1_size = fcl1_size, fcl2_size = fcl2_size)


    def get_action(self, state, step, track):
        """
        Return the best action in the passed state, according to the model
        in training. Noise added for exploration
        """
        frames = np.asarray(state["img"])

        if len(frames.shape) == 4:
            frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
        else:
            frames = np.array(cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY))

        # from PIL import Image
        # img = Image.fromarray(frames[0,...])
        # img.show()
        # input(frames[0].shape)

        del state["img"]
        # state[0] = angle
        # state[1] = speedX
        # state[2] = speedY
        # state[3] = speedZ
        # state[4,..22] = track
        # state[23] = trackPos
        # state[24,25,26,27] = wheelSpinVel

        #         {'angle': 0.004747795623217383, 'speedX': -0.006684910040348768, 'speedY': 0.041558898985385895, 'speedZ': -0.008202389813959599, 'track': array([0.0576535 , 0.3080025 , 0.24738051, 0.2081425 , 0.186822  ,
        #        0.1768675 , 0.17175949, 0.16740601, 0.1643635 , 0.161378  ,
        #        0.158448  , 0.1555745 , 0.151646  , 0.147292  , 0.1395125 ,
        #        0.125422  , 0.10596   , 0.0857255 , 0.0501905 ], dtype=float32), 'trackPos': 0.0025410999078303576, 'wheelSpinVel': array([ 0.     ,  0.     ,  2.31036, -2.40034], dtype=float32)}
        # tf.Tensor(
        # [[ 0.0047478  -0.00668491  0.0415589  -0.00820239  0.0576535   0.3080025
        #    0.24738051  0.2081425   0.186822    0.1768675   0.17175949  0.16740601
        #    0.1643635   0.161378    0.158448    0.1555745   0.151646    0.147292
        #    0.13951249  0.125422    0.10596     0.0857255   0.0501905   0.0025411
        #    0.          0.          2.31035995 -2.40034008]], shape=(1, 28), dtype=float64)

        composite_state = np.zeros(shape = (self.n_states))

        # standard sensors
        composite_state[1] = state["speedX"]
        composite_state[2] = state["speedY"]
        composite_state[3] = state["speedZ"]
        composite_state[24:] = state["wheelSpinVel"]


        frames = frames.reshape((self.img_height, self.img_width, self.stack_depth))

        frames = tf.expand_dims(frames, axis = 0)
        # use the estimator model to get the current state
        estimated_state = self.estimator_model.predict(frames)[0]

        # angle at sensors[0]
        # track at sensors[1:-1]
        # trackPos at sensors[:-1]


        composite_state[0] = estimated_state[0]
        composite_state[3:22] = estimated_state[1:-1]
        composite_state[24] = estimated_state[-1]

        #take only random actions for the first episode
        if(step > self.guided_steps):
            composite_state = tf.expand_dims(composite_state, axis = 0)
            action = self.actor.model.predict(composite_state)[0]
        else:
            #explore the action space quickly
            action = self.simple_controller(composite_state)

        noise = self._noise()
        action_p = action + noise
        #clip the resulting action with the bounds
        action_p = np.clip(action_p, self.lower_bound, self.upper_bound)
        return action_p

    def simple_controller(self, state):
        speedX = state[2]
        action = np.zeros(self.n_actions)
        # steer to corner
        steer = state[1] * 10
        # steer to center
        steer -= state[5] * .10

        accel = self.prev_accel

        if speedX < 0.6 - (steer * 50):
            accel += .01
        else:
            accel -= .01

        if accel > 0.2:
            accel = 0.2

        if speedX < 0.3:
            accel += 1 / (speedX + .1)

        action[0] = steer
        action[1] = accel

        self.prev_accel = accel

        return action


    def learn(self, i):
        """
        Fill the buffer up to the batch size, then train both networks with
        experience from the replay buffer.
        """
        actor_loss = 0
        if self._memory.isReady(self.batch_size):
            actor_loss = self.train_helper()
        return actor_loss

    def save_models(self):
        self.actor.model.save(save_dir + "/actor")
        self.actor.target_model.save(save_dir + "/actor_target")
        self.critic.model.save(save_dir + "/critic")
        self.critic.target_model.save(save_dir + "/critic_target")

    """
    Train helper methods
    train_helper
    train_critic
    train_actor
    get_q_targets  Q values to train the critic
    get_gradients  policy gradients to train the actor
    """

    def train_helper(self):
        # get experience batch
        states, actions, rewards, terminal, states_n = self._memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, dtype=tf.float32)
        states_n = tf.convert_to_tensor(states_n)

        # train the critic before the actor
        self.train_critic(states, actions, rewards, terminal, states_n)
        actor_loss = self.train_actor(states)
        #update the target models
        self.critic.update_target()
        self.actor.update_target()
        return actor_loss

    def train_critic(self, states, actions, rewards, terminal, states_n):
        """
        Use updated Q targets to train the critic network
        """
        # TODO cleaner code, ugly passing of actor target model
        self.critic.train(states, actions, rewards, terminal, states_n, self.actor.target_model, self.gamma)

    def train_actor(self, states):
        """
        Train the actor network with the critic evaluation
        """
        # TODO cleaner code, ugly passing of critic model
        return self.actor.train(states, self.critic.model)

    def remember(self, state, state_new, action, reward, terminal):
        """
        replay buffer interfate to the outsize
        """
        self._memory.remember(state, state_new, action, reward, terminal)
