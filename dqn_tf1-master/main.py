import tensorflow.compat.v1 as tf
import numpy as np

from config import Config
from dqn.agent import Agent
from dqn.environment import Environment

import matplotlib.pyplot as plt

def main():
    sess = tf.Session()
    config = Config()
    env = Environment(config)
    agent = Agent(sess, config, env)
    if config.test:
        agent.play(test=False)
    elif config.train:
        train_data = agent.train()
        plt.plot(train_data)
        plt.xlabel('Epochs')
        plt.ylabel('avg_ep_r')
        plt.title('Training avg_ep_r')
        plt.show()

    else:
        agent.play()


if __name__ == "__main__":
    main()
