import torch
torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import gym
import numpy as np
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig, CausalMnistBanditsEnv
from agents.baseline import UCBSocket, GaussianThompsonSocket, BaselineAgent
from argparse_dataclass import ArgumentParser

from utils.wb_vis import WBVis

from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s:%(filename)s:%(message)s',
                     datefmt='%m/%d %I:%M:%S %p',  
                     level=logging.WARNING)

logger = logging.getLogger(__name__)

@dataclass
class Options(CausalMnistBanditsConfig):
  seed: int = 5000
  log_every: int = 1 
  cuda:bool = False
  do_nothing: float = 0

if __name__ == '__main__':
    parser = ArgumentParser(Options)
    config = parser.parse_args()

    mnist_env = CausalMnistBanditsEnv()
    mnist_env.init(config)

    logger.warning(config)
    logger.warning(mnist_env)

    config.causal_ids = mnist_env.causal_ids
    agent = BaselineAgent(config.num_arms*2+1, UCBSocket)
    
    vis = WBVis(config, agent, mnist_env) if config.log_every > 0 else None
    timestep = mnist_env.reset()

    with tqdm(total=config.num_ts) as pbar:
        while not timestep.done:

            if config.log_every > 0 and timestep.id % config.log_every == 0:
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

    if config.log_every > 0: vis.finish()

