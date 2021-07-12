import gym
import numpy as np
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig
from agents import CmnistBanditAgent, AgentConfig
from argparse_dataclass import ArgumentParser

import logging
logging.basicConfig(format='%(asctime)s:%(filename)s:%(message)s',
                     datefmt='%m/%d %I:%M:%S %p',  
                     level=logging.DEBUG)

logger = logging.getLogger(__name__)


@dataclass
class Options(CausalMnistBanditsConfig, AgentConfig):
  seed: int = 5000
  debug: bool = False

  log_file: str = None



if __name__ == '__main__':
    parser = ArgumentParser(Options)
    config = parser.parse_args()
    
    mnist_env = gym.make('CausalMnistBanditsEnv-v0')
    mnist_env.init(config)

    logger.info(config)

    agent = CmnistBanditAgent(config)

    timestep = mnist_env.reset()

    while not timestep.done:

        if config.num_ts * config.do_nothing < timestep.id:
            op = agent.choose(timestep)
        else:
            op = mnist_env.noop

        old_timestep, timestep = mnist_env.step(op)
        logger.info(f'[{timestep.id}] timestep: \n\t{old_timestep}')
        agent.observe(old_timestep)
        agent.train()

    



