import gym
import numpy as np
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig
from agents import VariationalAgent, VariationalAgentConfig
from argparse_dataclass import ArgumentParser

import matplotlib.pyplot as plt
from utils.vis import Vis
import logging
logging.basicConfig(format='%(asctime)s:%(filename)s:%(message)s',
                     datefmt='%m/%d %I:%M:%S %p',  
                     level=logging.DEBUG)

logger = logging.getLogger(__name__)


@dataclass
class Options(CausalMnistBanditsConfig, VariationalAgentConfig):
  seed: int = 5000
  debug: bool = False

  log_file: str = None
  figure_dir: str = None




if __name__ == '__main__':
    parser = ArgumentParser(Options)
    config = parser.parse_args()
    
    mnist_env = gym.make('CausalMnistBanditsEnv-v0')
    mnist_env.init(config)

    logger.info(config)

    agent = VariationalAgent(config)
    vis = Vis(mnist_env.causal_ids)

    timestep = mnist_env.reset()

    while not timestep.done:

        # collect telemetry
        vis.collect(agent, mnist_env, timestep)

        # mnist_env.compute_kl(agent)
        if config.num_ts * config.do_nothing < timestep.id:
            op = agent.choose(timestep)
        else:
            op = mnist_env.noop

        old_timestep, timestep = mnist_env.step(op)
        agent.observe(old_timestep)
        agent.train()

        
    if config.figure_dir:
        vis.save_plots(config.figure_dir)


    



