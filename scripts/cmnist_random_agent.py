import gym
import agents
import numpy as np


import causal_env
from causal_env.envs import CausalMnistBanditsConfig

import argparse
from argparse_dataclass import ArgumentParser

SEED = 8888

import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    config = CausalMnistBanditsConfig(num_arms=10, causal_arms=3, num_ts=1000, seed=SEED)

    mnist_env = gym.make('CausalMnistBanditsEnv-v0')
    mnist_env.init(config)
    
    timestep = mnist_env.reset()

    while not timestep.done:
        action = mnist_env.action_space.sample()
        timestep = mnist_env.step(action)