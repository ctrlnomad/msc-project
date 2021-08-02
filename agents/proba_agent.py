from typing import Tuple, List
from dataclasses import dataclass, field
import numpy as np

from collections import deque
from functools import partial

import torch
import torch.nn as nn
from torch.nn.modules import utils
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.distributions as distributions

from causal_env.envs import Timestep, TimestepDataset
from agents.base_agent import BaseAgent

import agents.uncertainty_estimators.arches as arches
import agents.uncertainty_estimators.estimators as estimators

import utils
import utils.mnist 

import logging
logger = logging.getLogger(__name__)

@dataclass
class VariationalAgentConfig:
    Arch: nn.Module
    Estimator: estimators.BaseEstimator

    dim_in: Tuple[int] = (1, 28, 28)
    memsize: int = 100_000
    mc_samples: int = 100
    ensemble_size: int = 100

    do_nothing: float = 0.5 # do nothing for this proportion of steps
    cuda: bool = False
    batch_size:int = 32


class VariationalAgent(BaseAgent):
    """
    half the round we learn by empty interventions
    the other times we learn by choosing based on epsitemic uncertainty
    we evaluate by regret. 
    """
    def __init__(self, config: VariationalAgentConfig):
        # learning is the ITE of causal arms
        self.memory = deque(maxlen=config.memsize)

        make = partial(config.Arch, config)
        self.estimator = config.Estimator(make, config)
        self.config = config

        self.digit_sampler = utils.mnist.MnistSampler()
        
        if self.config.cuda:
            self.effect_estimator.cuda()

    def observe(self, timestep: Timestep):
        self.memory.append(timestep)

    def train(self, n_epochs:int=1):
        if len(self.memory) <= self.config.batch_size:
            logger.info('agent not training, not enough data')
            return 

        dataset = TimestepDataset(self.memory) # deconfound
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        for e in range(n_epochs):
            logger.info(f'[{e}] starting training ...')
            losses = self.estimator.train(loader)
            logger.info(f'[{e}] training finished; loss is at: [{self.history.loss[-1]:.4f}]')
        
    def act(self, timestep: Timestep):
        uncertaitnties = self.compute_digit_uncertainties(timestep.context)
        loc = (uncertaitnties.max() == uncertaitnties).nonzero().squeeze()
        intervention = loc[0] + len(timestep.context)* loc[1]
        return intervention.item()
    

    def compute_digit_uncertainties(self, contexts: torch.Tensor):
        variances = self.estimator.compute_uncertainty(contexts, self.config.mc_samples)
        return variances 

    def compute_digit_distributions(self, contexts: torch.Tensor):
        return self.estimator(contexts)

    def compute_best_action(self, contexts: torch.Tensor):
        mu, _ = self.estimator(contexts)
        return (mu.max() == mu).nonzero().squeeze()