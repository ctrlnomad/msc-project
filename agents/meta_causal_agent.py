from typing import Tuple, List
from dataclasses import dataclass
import numpy as np
from collections import deque

import torch
import torch.nn as nn

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

    def __init__(self, config: MetaCausalAgentConfig): # not quite variational agent config
        # learning is the ITE of causal arms
        self.config = config

        self.memory = deque(maxlen=config.memsize)

        self.estimator = estimators.StructEstimator(config)

    def observe(self, timestep: Timestep):
        self.memory.append(timestep)

    def train(self, n_epochs:int=1):
        if len(self.memory) <= self.config.batch_size:
            logger.info('agent not training, not enough data')
            return 

        dataset = TimestepDataset(self.memory)
        
        logger.info(f'starting training ...')
        loss, kl, ll = self.estimator.train(dataset, num_epochs=n_epochs)
        logger.info(f'training finished')
        return loss, kl, ll
        
    def act(self, timestep: Timestep):
        uncertaitnties = self.compute_digit_uncertainties(timestep.context)
        loc = (uncertaitnties.max() == uncertaitnties).nonzero().squeeze()
        intervention = loc[1] + len(timestep.context)* loc[0]
        return intervention.item()
    

    def compute_digit_uncertainties(self, contexts: torch.Tensor):
        variances = self.estimator.compute_uncertainty(contexts)
        return variances 

    def compute_digit_distributions(self, contexts: torch.Tensor):
        mus, sigmas = self.estimator(contexts)
        return mus, sigmas

    def compute_best_action(self, contexts: torch.Tensor):
        mu, _ = self.estimator(contexts)

        best_action = (mu.max() == mu).nonzero().squeeze()
        return best_action


    def compute_structural_entropy(self,):
        pass

    def compute_causal_contexts(self,):
        pass