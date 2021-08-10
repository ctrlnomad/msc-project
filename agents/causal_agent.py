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
class CausalAgentConfig:
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

    causal_ids: List[int] = None
    
class CausalAgent(BaseAgent):
    """
    half the round we learn by empty interventions
    the other times we learn by choosing based on epsitemic uncertainty
    we evaluate by regret. 
    """
    def __init__(self, config: CausalAgentConfig): # not quite variational agent config
        # learning is the ITE of causal arms
        assert config.causal_ids is not None
        self.config = config

        self.memory = deque(maxlen=config.memsize)

        self.estimator = config.Estimator(config.Arch, config, self.make_decounfound())
        

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
            logger.info(f'[{e}] training finished') # TODO print losss
        
    def act(self, timestep: Timestep):
        uncertaitnties = self.compute_digit_uncertainties(timestep.context)
        loc = (uncertaitnties.max() == uncertaitnties).nonzero().squeeze()
        intervention = loc[1] + len(timestep.context)* loc[0]
        return intervention.item()
    

    def compute_digit_uncertainties(self, contexts: torch.Tensor):
        variances = self.estimator.compute_uncertainty(contexts)
        return variances 

    def compute_digit_distributions(self, contexts: torch.Tensor):
        return self.estimator(contexts)

    def compute_best_action(self, contexts: torch.Tensor):
        mu, _ = self.estimator(contexts)

        best_action = (mu.max() == mu).nonzero().squeeze()
        return best_action

    
    def make_decounfound(self):

        def deconfound(model, contexts, treatments,  effects):
            with torch.no_grad():
                print('deconfounding')
                causal_treatments = treatments.index_select(dim=1, index=self.config.causal_ids)
                bs = len(contexts)

                confounding_contexts = contexts.index_select(dim=1, index=self.config.causal_ids).view(-1, *self.config.dim_in)
                mu_pred, _ = model(confounding_contexts)
                mu_pred = torch.stack(mu_pred).view(bs, -1, 2) # what
                # we know how many causal effects there are 4 in this case
                num_causal_arms = len(self.config.causal_ids)
                deconfounded_effects  = torch.zeros(bs, num_causal_arms)

                mu_pred = mu_pred.gather(-1, causal_treatments[..., None]).squeeze()
                for i in range(bs):
                    deconfounded_effects[i] = effects[i] - (mu_pred[i].sum() - mu_pred[i])

                deconfounded_effects = deconfounded_effects.flatten()
                causal_treatments = causal_treatments.flatten()
                return confounding_contexts, causal_treatments, deconfounded_effects

        return deconfound