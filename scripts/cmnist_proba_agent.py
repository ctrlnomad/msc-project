import torch
torch.autograd.set_detect_anomaly(True)


import gym
import numpy as np
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig
from agents import VariationalAgent, VariationalAgentConfig
from argparse_dataclass import ArgumentParser

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

  telemetry_every: int = 1 


import agents.uncertainty_estimators.estimators as estimators
import agents.uncertainty_estimators.arches as arches

if __name__ == '__main__':
    parser = ArgumentParser(Options)
    config = parser.parse_args()

    config.Arch = arches.ConvNet
    config.Estimator = estimators.DropoutEstimator

    logger.warn(f'running with Arch={config.Arch} and Estimator={config.Estimator}')

    mnist_env = gym.make('CausalMnistBanditsEnv-v0')
    mnist_env.init(config)

    logger.info(config)

    agent = VariationalAgent(config)

    timestep = mnist_env.reset()

    while not timestep.done:

        # mnist_env.compute_kl(agent)
        if config.num_ts * config.do_nothing < timestep.id:
            op = agent.act(timestep)
        else:
            op = mnist_env.noop

        old_timestep, timestep = mnist_env.step(op)
        agent.observe(old_timestep)
        agent.train()

    


    



