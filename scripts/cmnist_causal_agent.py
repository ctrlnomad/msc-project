import gym
import numpy as np
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig, Timestep
from agents import MnistBanditAgent, AgentConfig


import logging
logger = logging.getLogger(__name__)


@dataclass
class Options(CausalMnistBanditsConfig, AgentConfig):
  log_file: str 
  cuda: bool = True
  
  seed: int = 5000
  debug: bool = False


from argparse_dataclass import ArgumentParser, parse_args



if __name__ == '__main__':
    parser = ArgumentParser(Options)
    config = parser.parse_args()
    
    mnist_env = gym.make('CausalMnistBanditsEnv-v0')
    mnist_env.init(config)
    
    agent = MnistBanditAgent()

    obs = mnist_env.reset()
    done = False

    while not done:
        print('new step')
        action = mnist_env.action_space.sample()
        timestep = mnist_env.step(action)

        print(config)
        print(mnist_env.digit_ITEs)
        print(ctx.shape)
        break