import gym
import numpy as np
from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig
from agents import CmnistBanditAgent, AgentConfig
from argparse_dataclass import ArgumentParser

from utils.vis import Vis
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
    vis = Vis(mnist_env.causal_ids)

    timestep = mnist_env.reset()

    while not timestep.done:

        # collect telemetry
        vis.collect(agent, mnist_env)

        # mnist_env.compute_kl(agent)
        if config.num_ts * config.do_nothing < timestep.id:
            op = agent.choose(timestep)
        else:
            op = mnist_env.noop

        old_timestep, timestep = mnist_env.step(op)
        agent.observe(old_timestep)
        agent.train()

        

    fig = vis.plot_loss()
    fig.show()
    fig = vis.plot_uncerstainty()
    fig.show()
    fig = vis.plot_kl()
    fig.show()
    fig = vis.plot_regret()
    fig.show()
    



