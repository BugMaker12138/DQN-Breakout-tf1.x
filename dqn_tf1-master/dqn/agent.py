import numpy as np
import tensorflow.compat.v1 as tf

import random

import time
import cv2

from .model import DQN
from .replay_memory import ReplayMemory
from .short_term_memory import ShortTermMemory

from .utils import makedir_if_there_is_no

from tqdm import tqdm


class Agent(DQN):

    def __init__(self, sess, config, env):
        super(Agent, self).__init__(config)
        self.sess = sess
        self.env = env
        self.memory = ReplayMemory(config)
        self.short_term = ShortTermMemory(config)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.get_summary_ops()
        self._build_model()
        self.make_directories()

    def make_directories(self):
        dirs = [self.config.mem_dir, self.config.log_dir, self.config.ckpt_dir]
        for d in dirs:
            makedir_if_there_is_no(d)

    def get_summary_ops(self):
        with tf.variable_scope("summary"):
            summary_tags = ["average.reward", "average.loss", "average.q", \
                            "episode.avg_reward", "episode.max_reward", \
                            "training.learning_rate"]
            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in summary_tags:
                self.summary_placeholders[tag] = tf.placeholder(
                    tf.float32, None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(
                    "{}/{}".format(self.config.env_name, tag),
                    self.summary_placeholders[tag])

    def summarize(self, tag_dict, step=None):
        summary_lst = self.sess.run(
            [self.summary_ops[tag] for tag in tag_dict.keys()], {
                self.summary_placeholders[tag]: value
                for tag, value in tag_dict.items()
            })
        for elt in summary_lst:
            self.writer.add_summary(elt, step)
        self.writer.flush()
        print("\n****summarized")

    def train(self):
        a = []
        self.make_directories()
        if self.config.load_ckpt:
            self.load()

        screen, action, reward, done = self.env.start_randomly()

        for _ in range(self.config.history_length):
            self.short_term.add(screen)

        start_step = self.sess.run(self.global_step) + 1
        for self.step in tqdm(
                range(start_step, self.config.max_step),
                ncols=70,
                total=self.config.max_step,
                initial=start_step):
            if self.step == start_step or self.step == self.config.replay_start_size:
                self.update_cnt = 0
                ep_reward = 0.
                total_reward = 0.
                ep_rewards = []
                max_avg_record = 0.
                self.total_loss = 0.
                self.total_q = 0.

            action,eps = self.choose_action()
            screen, reward, done = self.env.act(action)
            self.after_act(action, screen, reward, done)

            if done:
                screen, action, reward, done = self.env.start_randomly()
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            total_reward += reward

            if self.step > self.config.replay_start_size:
                if self.step % self.config.summarize_step == 0:
                    avg_reward = total_reward / self.config.summarize_step
                    avg_loss = self.total_loss / self.update_cnt
                    avg_q = self.total_q / self.update_cnt

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward = 0
                        avg_ep_reward = 0
# 加上epsolin
                    a.append(avg_ep_reward)
                    print(
                        "\navg_r: {:.4f}, avg_l: {:.6f}, avg_q: {:3.6f}, avg_ep_r: {:.4f}, max_ep_r: {:.4f},epsilon:{:4f}"
                        .format(avg_reward, avg_loss, avg_q, avg_ep_reward,
                                max_ep_reward,eps))

                    if max_avg_record * 0.9 <= avg_ep_reward:
                        self.sess.run(self.global_step.assign(self.step))
                        self.save(self.step)
                        max_avg_record = max(max_avg_record, avg_ep_reward)

                    summary_dict = {
                        "average.reward":
                        avg_reward,
                        "average.loss":
                        avg_loss,
                        "average.q":
                        avg_q,
                        "episode.max_reward":
                        max_ep_reward,
                        "episode.avg_reward":
                        avg_ep_reward,
                        "training.learning_rate":
                        self.sess.run(self.decayed_lr,
                                      {self.lr_step: self.step})
                    }
                    self.summarize(summary_dict, self.step)
                    self.update_cnt = 0
                    ep_reward = 0.
                    total_reward = 0.
                    ep_rewards = []
                    self.total_loss = 0.
                    self.total_q = 0.
        return a

    def get_eps(self):
        if not self.config.train:
            return self.config.test_exploration
        elif self.step < self.config.replay_start_size:
            return 1.
        elif self.step < self.config.final_exploration_step*10:
            return 1 - 0.9 * ((0.1*self.step - self.config.replay_start_size) /
                              (self.config.final_exploration_step -
                               self.config.replay_start_size))
        else:
            return self.config.final_exploration

    def choose_action(self):
        eps = self.get_eps()
        if random.random() < eps:
            a = random.randrange(self.env.action_space_size)
        else:
            a = self.sess.run(self.greedy_action,
                              {self.S_in: [self.short_term.frames]})[0]
        return a,eps

    def reward_clipping(self, r):
        return max(self.config.min_reward, min(self.config.max_reward, r))

    def calc_targets(self, R, NS, Done):
        NQ = self.sess.run(self.fixed_Q, {self.fixed_S_in: NS})
        max_NQ = np.max(NQ, axis=1)
        return (1. - Done) * self.config.df * max_NQ + R

    def experience_replay(self):
        S, A, R, NS, Done = self.memory.get_batch()
        targets = self.calc_targets(R, NS, Done)
        _, Q, loss = self.sess.run(
            [self.train_op, self.Q, self.loss], {
                self.S_in: S,
                self.A_in: A,
                self.target: targets,
                self.lr_step: self.step
            })
        self.total_loss += loss
        self.total_q += np.mean(Q)
        self.update_cnt += 1

    def after_act(self, action, screen, reward, done):
        reward = self.reward_clipping(reward)

        self.short_term.add(screen)
        self.memory.memorize(action, screen, reward, done)

        if self.step > self.config.replay_start_size:
            if self.step % self.config.replay_frequency == 0:
                self.experience_replay()
            if self.step % self.config.fixed_net_update_frequency == 0:
                self.update_fixed_target()

    def play(self, test=False):
        try:
            self.load()
        except:
            print("****FAILED to load weights. Can't play anymore")
            return None

        screen, action, reward, done = self.env.initialize_game()

        #####


        #####3
        for _ in range(self.config.history_length):
            self.short_term.add(screen)

        ep_rewards = []
        count = 0
        for i in range(self.config.test_play_num):
            ep_reward = 0
            while not done:
                if test:
                    action = int(
                        input("Enter the action (0 ~ {}): ".format(
                            self.env.action_space_size - 1)))
                else:
                    action,_ = self.choose_action()
                screen, reward, done = self.env.act(action)
                time.sleep(1 / 240)
                self.short_term.add(screen)
                ep_reward += reward
            ep_rewards.append(ep_reward)
            print("game #: {}, reward: {}".format(i + 1, ep_reward))
            self.memory.testsave()
            screen, action, reward, done = self.env.initialize_game()

        print("Evaluation Done.\n mean reward: {}, max reward: {}".format(
            np.mean(ep_rewards), np.max(ep_rewards)))



