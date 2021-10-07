import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import torch
torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from typing import Any
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig, CausalMnistBanditsEnv
from agents import ATEAgent, ATEAgentConfig

from simple_parsing import ArgumentParser

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
  api_key:str = ''
  log_every: int = 1

  random_explore: bool = False
  group: str = ''

import agents.uncertainty_estimators.estimators as estimators
import agents.uncertainty_estimators.arches as arches



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest='options')
    config = parser.parse_args().options

    config.Arch = arches.ConvNet
    config.Estimator = estimators.DropoutEstimator

    logger.warning(f'running with Arch={config.Arch} and Estimator={config.Estimator}')

    mnist_env = CausalMnistBanditsEnv()
    mnist_env.init(config)

    logger.warning(config)
    logger.warning(mnist_env)

    agent = ATEAgent(config)
    
    vis = WBVis(config, agent, mnist_env) if config.log_every > 0 else None
    timestep = mnist_env.reset()

    with tqdm(total=config.num_ts) as pbar:
        while not timestep.done:

            if config.log_every > 0 and timestep.id % config.log_every == 0:
                vis.collect(agent, mnist_env, timestep)
                vis.collect_arm_distributions(agent, mnist_env, timestep)

            op = mnist_env.noop
            if config.num_ts * config.do_nothing < timestep.id:
                if config.random_explore:
                    while op == mnist_env.noop:
                        op = mnist_env.action_space.sample()
                else:
                    op = agent.act(timestep)

            old_timestep, timestep = mnist_env.step(op)

            agent.observe(old_timestep)
            agent.train()

            pbar.update(1)

    if config.log_every > 0: vis.finish()

