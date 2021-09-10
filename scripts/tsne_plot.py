import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import torch, numpy as np
torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from dataclasses import dataclass

from causal_env.envs import CausalMnistBanditsConfig, CausalMnistBanditsEnv, causal_mnist
from agents import ATEAgent, ATEAgentConfig
from sklearn.manifold import TSNE

from simple_parsing import ArgumentParser

import utils
from utils import mnist
from utils.wb_vis import WBVis

import matplotlib.pyplot as  plt
from pathlib import Path
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

  tsne_dir: str = './tsne_plots'
  telemetry_every: int = 1

  perplexity: int =  30
  tsne_every: int = 500

  random_explore: bool = False


import agents.uncertainty_estimators.estimators as estimators
import agents.uncertainty_estimators.arches as arches


def compute_tsne_embeddings(config, agent, env, file_name):
    
    data, targets = env.digit_sampler.sample_digits(config.num_arms, n_samples=1_000)
    
    if config.cuda:
        data = data.cuda()

    targets = utils.safenumpy(targets).astype(int)

    causal_ids = utils.safenumpy(torch.nonzero(env.ite[1,:], as_tuple=True)[0])

    emb = utils.safenumpy(agent.estimator.net.conv_block(data))
    logger.warning('starting TSNE ...')
    tsne_emb = TSNE(n_components=2, perplexity=config.perplexity, random_state=config.seed).fit_transform(emb)
    logger.warning('done TSNE, plotting ...')

    fig, ax = plt.subplots(figsize=(12,10))
    plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=targets, cmap='Spectral')

    plt.setp(ax, xticks=[],  yticks=[])
    bar = plt.colorbar(boundaries=np.arange(config.num_arms+1)-0.5)
    bar.set_ticks(np.arange(config.num_arms))
    bar.set_ticklabels(np.arange(config.num_arms))
    plt.title(f'Causal digits are {list(causal_ids)}')
    plt.savefig(file_name)
    logger.warning(f'done plotting, saved to {file_name}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest='options')
    config = parser.parse_args().options

    config.Arch = arches.ConvNet
    config.Estimator = estimators.DropoutEstimator

    tsne_dir = Path(config.tsne_dir)
    tsne_dir.mkdir(exist_ok = True)

    logger.warning(f'running with Arch={config.Arch} and Estimator={config.Estimator}')

    mnist_env = CausalMnistBanditsEnv()
    mnist_env.init(config)

    logger.warning(config)
    logger.warning(mnist_env)

    config.causal_ids = mnist_env.causal_ids
    agent = ATEAgent(config)

    timestep = mnist_env.reset()

    compute_tsne_embeddings(config, agent, mnist_env, tsne_dir / 'Before Training')

    with tqdm(total=config.num_ts) as pbar:
        while not timestep.done:

            if timestep.id % config.tsne_every == 0:
                compute_tsne_embeddings(config, agent, mnist_env, tsne_dir / f'At Timestep #{timestep.id }')

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

    compute_tsne_embeddings(config, agent, mnist_env,tsne_dir /   'After Training')
