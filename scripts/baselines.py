import gym
import numpy as np
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig, CausalMnistBanditsEnv
from agents.baseline import BaselineAgent, UCBSocket, GaussianThompsonSocket
from simple_parsing import ArgumentParser

import utils
from utils.wb_vis import WBVis
import logging

logging.basicConfig(format='%(asctime)s:%(filename)s:%(message)s',
                     datefmt='%m/%d %I:%M:%S %p',  
                     level=logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class Options(CausalMnistBanditsConfig):
  seed: int = 5000
  telemetry_every: int = 1 
  cuda: bool = False
  api_key: str = ''
  group: str = ''
  socket:str = 'thompson'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest='options')
    config = parser.parse_args().options

    if config.socket == 'thompson':
      socket = GaussianThompsonSocket
    elif config.socket =='ucb':
      socket = UCBSocket

    logger.warn(f'FILE: {__name__} \n\trunning with Socket={socket} for T={config.num_ts}')
    
    mnist_env = CausalMnistBanditsEnv()
    mnist_env.init(config)

    logger.info(config)

    agent = BaselineAgent(mnist_env.action_space.n, socket)
    vis = WBVis(config, agent, mnist_env)

    timestep = mnist_env.reset()

    while not timestep.done:

      # collect telemetry
      if timestep.id % config.telemetry_every == 0:
          vis.collect(agent, mnist_env, timestep)
          vis.collect_arm_distributions(agent, mnist_env, timestep)

      op = agent.act(timestep)

      old_timestep, timestep = mnist_env.step(op)
      agent.observe(old_timestep)
    
    print('\n\n', mnist_env.ite, mnist_env.variance)



  



