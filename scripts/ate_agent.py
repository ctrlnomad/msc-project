import torch
torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import gym
import numpy as np
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig
from agents import ATEAgent, ATEAgentConfig
from argparse_dataclass import ArgumentParser

from utils.wb_vis import WBVis

from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s:%(filename)s:%(message)s',
                     datefmt='%m/%d %I:%M:%S %p',  
                     level=logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class Options(CausalMnistBanditsConfig, ATEAgentConfig):
  seed: int = 5000
  debug: bool = False

  log_file: str = None

  telemetry_dir: str = None
  telemetry_every: int = 1 


import agents.uncertainty_estimators.estimators as estimators
import agents.uncertainty_estimators.arches as arches



if __name__ == '__main__':
    parser = ArgumentParser(Options)
    config = parser.parse_args()

    config.Arch = arches.ConvNet
    config.Estimator = estimators.DropoutEstimator

    logger.warning(f'running with Arch={config.Arch} and Estimator={config.Estimator}')

    mnist_env = gym.make('CausalMnistBanditsEnv-v0')
    mnist_env.init(config)

    logger.warning(config)
    logger.warning(mnist_env)

    agent = ATEAgent(config)
    vis = WBVis(config)

    timestep = mnist_env.reset()

    with tqdm(total=config.num_ts) as pbar:
        while not timestep.done:

            if timestep.id % config.telemetry_every == 0:
                vis.collect(agent, mnist_env, timestep)
                vis.collect_arm_distributions(agent, mnist_env, timestep)

            if config.num_ts * config.do_nothing < timestep.id:
                op = agent.act(timestep)
            else:
                op = mnist_env.noop

            old_timestep, timestep = mnist_env.step(op)

            agent.observe(old_timestep)
            agent.train()

            pbar.update(1)

    vis.finish()

    



