import gym
import numpy as np
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig
from agents import MnistBanditAgent, AgentConfig


import logging
logging.basicConfig(filename='example.log', format='%(asctime)s:%(filename)s:%(message)s',
                     datefmt='%m/%d %I:%M:%S %p',  level=logging.DEBUG)

logger = logging.getLogger(__name__)


@dataclass
class Options(CausalMnistBanditsConfig, AgentConfig):
  cuda: bool = True

  seed: int = 5000
  debug: bool = False

  log_file: str = None


from argparse_dataclass import ArgumentParser, parse_args



if __name__ == '__main__':
    parser = ArgumentParser(Options)
    config = parser.parse_args()
    
    mnist_env = gym.make('CausalMnistBanditsEnv-v0')
    mnist_env.init(config)

    logger.info(config)

    agent = MnistBanditAgent(config)

    timestep = mnist_env.reset()
    done = False

    while not done:
        agent.observe(timestep)
        agent.train()

        if config.num_ts * config.do_nothing >= timestep.id:
            op = agent.choose(timestep)
        else:
            op = mnist_env.noop

        timestep = mnist_env.step(op)

        done = timestep.done
    



