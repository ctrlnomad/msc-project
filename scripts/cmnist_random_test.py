import gym
import agents
import numpy as np


import causal_env
from causal_env.envs import CausalMnistBanditsConfig

import argparse

SEED = 8888

import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    config = CausalMnistBanditsConfig(num_arms=10, causal_arms=3, num_ts=1000, seed=SEED)

    mnist_env = gym.make('CausalMnistBanditsEnv-v0')
    mnist_env.init(config)
    
    obs = mnist_env.reset()
    done = False

    while not done:
        print('new step')
        action = mnist_env.action_space.sample()
        ctx, reward, done, treatments = mnist_env.step(action)

        print(config)
        print(mnist_env.digit_ITEs)
        print(ctx.shape)
        break