import gym
import numpy as np
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig
from agents.baseline import BaselineAgent, UCBSocket, GaussianThompsonSocket
from argparse_dataclass import ArgumentParser

from utils.tb_vis import TensorBoardVis
import logging

logging.basicConfig(format='%(asctime)s:%(filename)s:%(message)s',
                     datefmt='%m/%d %I:%M:%S %p',  
                     level=logging.DEBUG)

logger = logging.getLogger(__name__)


@dataclass
class Options(CausalMnistBanditsConfig):
  seed: int = 5000
  debug: bool = False

  log_file: str = None

  telemetry_dir: str = None
  telemetry_every: int = 1 


if __name__ == '__main__':
    parser = ArgumentParser(Options)
    config = parser.parse_args()
    socket = GaussianThompsonSocket
    logger.warn(f'FILE: {__name__} \n\trunning with Socket={socket} for T={config.num_ts}')
    
    mnist_env = gym.make('CausalMnistBanditsEnv-v0')
    mnist_env.init(config)

    logger.info(config)

    agent = BaselineAgent(mnist_env.action_space.n, socket)
    vis = TensorBoardVis(config)

    timestep = mnist_env.reset()

    while not timestep.done:

        # collect telemetry
        if timestep.id % config.telemetry_every == 0:
            vis.collect(agent, mnist_env, timestep)

        op = agent.act(timestep)

        old_timestep, timestep = mnist_env.step(op)
        agent.observe(old_timestep)



    


    



