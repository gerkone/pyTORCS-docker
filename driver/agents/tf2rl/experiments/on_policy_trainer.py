import os
import time

import numpy as np
import tensorflow as tf

from cpprb import ReplayBuffer

from agents.tf2rl.experiments.trainer import Trainer
from agents.tf2rl.experiments.utils import save_path, frames_to_gif
from agents.tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from agents.tf2rl.misc.discount_cumsum import discount_cumsum


def unpack_state(state):
    """
    state dict to state array if fixed order
    """
    state_array = np.zeros(28)

    state_array[0] = state["speedX"]
    state_array[1] = state["speedY"]
    state_array[2] = state["speedZ"]
    state_array[3] = state["angle"]
    state_array[4] = state["trackPos"]
    state_array[5:9] = state["wheelSpinVel"]
    state_array[9:28] = state["track"]

    return state_array



class OnPolicyTrainer(Trainer):
    def __init__(self, policy, env, args, test_env = None, expert_trajs = None):
        super().__init__(policy, env, args, test_env)
        self.expert_trajs = expert_trajs

    def __call__(self, track_list):
        # Prepare buffer
        self.replay_buffer = get_replay_buffer(
            self._policy, self._env)
        kwargs_local_buf = get_default_rb_dict(
            size=self._policy.horizon, env=self._env)
        kwargs_local_buf["env_dict"]["logp"] = {}
        kwargs_local_buf["env_dict"]["val"] = {}

        self.local_buffer = ReplayBuffer(**kwargs_local_buf)
        if self.expert_trajs != None:
            expert_trajs_size = self.expert_trajs["action"].shape[0]
            exp_i = 0

        episode_steps = 0
        episode_return = 0
        episode_start_time = time.time()
        n_episode = 0


        for track in track_list:
            total_steps = np.array(0, dtype = np.int32)
            tf.summary.experimental.set_step(total_steps)
            self._env.set_track(track)
            while total_steps < self._max_steps:
                if self.expert_trajs != None and n_episode % 100 == 0:
                    self.logger.info("Training on expert data")
                    for i in range(self._policy.horizon):
                        # use expert data. try to vary agent behaviour
                        act = self.expert_trajs["action"][(exp_i + i) % expert_trajs_size]
                        obs = self.expert_trajs["state"][(exp_i + i) % expert_trajs_size]
                        next_obs = self.expert_trajs["state_new"][(exp_i + i) % expert_trajs_size]
                        reward = self.expert_trajs["reward"][(exp_i + i) % expert_trajs_size]
                        # certain action
                        logp = np.log(1)
                        _, _, val = self._policy.get_action_and_val(obs)
                        # sum of rewards for a whole episode
                        val = discount_cumsum(self.expert_trajs["reward"][(exp_i + i + 1) % expert_trajs_size:
                                        (exp_i + i + self._policy.horizon) % expert_trajs_size], self._policy.discount)[:-1]
                        val = val[0]
                        self.local_buffer.add(
                            obs = obs, act = act, next_obs = next_obs,
                            rew = reward, done = False, logp = logp, val = val)
                    exp_i += self._policy.horizon
                    n_episode += 1
                else:
                    if total_steps <= 0:
                        obs = self._env.reset()
                        obs = unpack_state(obs)
                    # collect samples
                    for _ in range(self._policy.horizon):
                        if self._normalize_obs:
                            obs = self._obs_normalizer(obs, update=False)
                        # individual_noise to sample actions differently
                        act, logp, val = self._policy.get_action_and_val(obs, individual_noise = False)
                        env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
                        next_obs, reward, done = self._env.step(env_act)
                        next_obs = unpack_state(next_obs)

                        episode_steps += 1
                        total_steps += 1
                        episode_return += reward

                        done_flag = done
                        if (hasattr(self._env, "_max_episode_steps") and
                            episode_steps == self._env._max_episode_steps):
                            done_flag = False
                        self.local_buffer.add(
                            obs=obs, act=act, next_obs=next_obs,
                            rew=reward, done=done_flag, logp=logp, val=val)
                        obs = next_obs

                        if done or episode_steps == self._episode_max_steps:
                            tf.summary.experimental.set_step(total_steps)
                            self.finish_horizon()
                            obs = self._env.reset()
                            obs = unpack_state(obs)

                            n_episode += 1
                            fps = episode_steps / (time.time() - episode_start_time)
                            self.logger.info(
                                "Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                                    n_episode, int(total_steps), episode_steps, episode_return, fps))
                            tf.summary.scalar(name="Common/training_return", data=episode_return)
                            tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)
                            tf.summary.scalar(name="Common/fps", data=fps)
                            episode_steps = 0
                            episode_return = 0
                            episode_start_time = time.time()

                        # if total_steps % self._test_interval == 0:
                        #     avg_test_return, avg_test_steps = self.evaluate_policy(total_steps)
                        #     self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                        #         total_steps, avg_test_return, self._test_episodes))
                        #     tf.summary.scalar(
                        #         name="Common/average_test_return", data=avg_test_return)
                        #     tf.summary.scalar(
                        #         name="Common/average_test_episode_length", data=avg_test_steps)
                        #     self.writer.flush()
                        #
                        if total_steps % self._save_model_interval == 0:
                            self.checkpoint_manager.save()

                self.finish_horizon(last_val=val)

                tf.summary.experimental.set_step(total_steps)

                # Train actor critic
                if self._policy.normalize_adv:
                    samples = self.replay_buffer.get_all_transitions()
                    mean_adv = np.mean(samples["adv"])
                    std_adv = np.std(samples["adv"])
                    # Update normalizer
                    if self._normalize_obs:
                        self._obs_normalizer.experience(samples["obs"])

                actor_loss = 0
                critic_loss = 0

                with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                    for _ in range(self._policy.n_epoch):
                        samples = self.replay_buffer._encode_sample(
                            np.random.permutation(self._policy.horizon))
                        if self._normalize_obs:
                            samples["obs"] = self._obs_normalizer(samples["obs"], update=False)
                        if self._policy.normalize_adv:
                            adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)
                        else:
                            adv = samples["adv"]
                        for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                            target = slice(idx * self._policy.batch_size,
                                           (idx + 1) * self._policy.batch_size)
                            al, cl = self._policy.train(
                                states=samples["obs"][target],
                                actions=samples["act"][target],
                                advantages=adv[target],
                                logp_olds=samples["logp"][target],
                                returns=samples["ret"][target])
                            actor_loss += al
                            critic_loss += cl
                    actor_loss = actor_loss / (int(self._policy.horizon / self._policy.batch_size) * self._policy.n_epoch)
                    critic_loss = critic_loss / (int(self._policy.horizon / self._policy.batch_size) * self._policy.n_epoch)
                    self.logger.info("Done {} epochs. Average actor loss {}".format(
                            self._policy.n_epoch, actor_loss, critic_loss))

        tf.summary.flush()

    def finish_horizon(self, last_val=0):
        self.local_buffer.on_episode_end()
        samples = self.local_buffer._encode_sample(
            np.arange(self.local_buffer.get_stored_size()))
        rews = np.append(samples["rew"], last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]
        if self._policy.enable_gae:
            advs = discount_cumsum(deltas, self._policy.discount * self._policy.lam)
        else:
            advs = deltas

        # Rewards-to-go, to be targets for the value function
        rets = discount_cumsum(rews, self._policy.discount)[:-1]
        self.replay_buffer.add(
            obs=samples["obs"], act=samples["act"], done=samples["done"],
            ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))
        self.local_buffer.clear()

    def evaluate_policy(self, total_steps):
        avg_test_return = 0.
        avg_test_steps = 0
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)
        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []
            obs = self._test_env.reset()
            obs = unpack_state(obs)

            avg_test_steps += 1
            for _ in range(self._episode_max_steps):
                if self._normalize_obs:
                    obs = self._obs_normalizer(obs, update=False)
                act, _ = self._policy.get_action(obs, test=True)
                act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
                next_obs, reward, done = self._test_env.step(act)
                next_obs = unpack_state(next_obs)

                avg_test_steps += 1
                if self._save_test_path:
                    replay_buffer.add(
                        obs=obs, act=act, next_obs=next_obs,
                        rew=reward, done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))
                elif self._show_test_progress:
                    self._test_env.render()
                episode_return += reward
                obs = next_obs
                if done:
                    break
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
                total_steps, i, episode_return)
            if self._save_test_path:
                save_path(replay_buffer.sample(self._episode_max_steps),
                          os.path.join(self._output_dir, prefix + ".pkl"))
                replay_buffer.clear()
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            avg_test_return += episode_return
        if self._show_test_images:
            images = tf.cast(
                tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
                tf.uint8)
            tf.summary.image('train/input_img', images, )
        return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes
