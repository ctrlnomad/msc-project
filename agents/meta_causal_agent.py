from typing import Tuple, List
from dataclasses import dataclass, field
import numpy as np
from collections import deque
from numpy.core.fromnumeric import nonzero

import torch
import torch.nn as nn
from torch.nn.modules import utils
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.distributions as distributions

from causal_env.envs import Timestep, TimestepDataset
from agents.base_agent import BaseAgent

import  agents.uncertainty_estimators.estimators as estimators

import logging
logger = logging.getLogger(__name__)


@dataclass
class MetaCausalAgentConfig:
    Arch: nn.Module = None
    Estimator: estimators.BaseEstimator = None

    dim_in: Tuple[int] = (1, 28, 28)
    memsize: int = 100_000
    mc_samples: int = 100
    ensemble_size: int = 10
    dropout_rate: float = 0.5

    do_nothing: float = 0.5 # do nothing for this proportion of steps
    cuda: bool = False
    batch_size:int = 32

    fixed_sigma: bool = False
    causal_ids: List[int] = None
    
class MetaCausalAgent(BaseAgent):
    """
    Works for one causal arm
    """
    def __init__(self, config: MetaCausalAgentConfig): # not quite variational agent config
        # learning is the ITE of causal arms
        self.config = config

        self.memory = deque(maxlen=config.memsize)

        self.estimator = estimators.MetaCausalEstimator(config)

    def observe(self, timestep: Timestep):
        self.memory.append(timestep)

    def train(self, n_epochs:int=1):
        if len(self.memory) <= self.config.batch_size:
            logger.info('agent not training, not enough data')
            return 

        dataset = TimestepDataset(self.memory) # deconfound
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        # we need to calculate the log likelihoods over the ten models
        for e in range(n_epochs):
            logger.info(f'[{e}] starting training ...')
            self.estimator.train(loader)
            logger.info(f'[{e}] training finished')


        
    def act(self, timestep: Timestep):
        uncertaitnties = self.compute_digit_uncertainties(timestep.context)
        loc = (uncertaitnties.max() == uncertaitnties).nonzero().squeeze()
        intervention = loc[1] + len(timestep.context)* loc[0]
        return intervention.item()
    

    def compute_digit_uncertainties(self, contexts: torch.Tensor):
        variances = self.estimator.compute_uncertainty(contexts)
        return variances 

    def compute_digit_distributions(self, contexts: torch.Tensor):
        mus, sigmas, _ = self.estimator(contexts)
        return mus, sigmas

    def compute_best_action(self, contexts: torch.Tensor):
        mu, _, _= self.estimator(contexts)

        best_action = (mu.max() == mu).nonzero().squeeze()
        return best_action

