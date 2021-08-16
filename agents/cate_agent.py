
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
class CATEAgentConfig:
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
    
class CATEAgent(BaseAgent):
    """
    half the round we learn by empty interventions
    the other times we learn by choosing based on epsitemic uncertainty
    we evaluate by regret. 
    """
    def __init__(self, config: CATEAgentConfig): # not quite variational agent config
        # learning is the ITE of causal arms
        assert config.causal_ids is not None
        self.config = config

        self.memory = deque(maxlen=config.memsize)

        self.estimator = config.Estimator(config.Arch, config, self.make_decounfound(), CATE=True)
        

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
        uncertaitnties = self.estimator.compute_uncertainty(timestep.context)
        loc = (uncertaitnties.max() == uncertaitnties).nonzero().squeeze()
        intervention = loc[1] + len(timestep.context)* loc[0]
        return intervention.item()
    

    def compute_digit_uncertainties(self, contexts: torch.Tensor):
        variances = self.estimator.compute_cate_uncertainty(contexts)
        return variances 

    def compute_digit_distributions(self, contexts: torch.Tensor):
        return self.estimator(contexts)

    def compute_best_action(self, contexts: torch.Tensor):
        mu, _ = self.estimator(contexts)

        best_action = (mu.max() == mu).nonzero().squeeze()
        return best_action

    
    def make_decounfound(self): 

        def deconfound(contexts, treatments,  ts_causal_ids, effects):

            with torch.no_grad():
                bs = len(contexts)

                causal_idxs = torch.nonzero(ts_causal_ids, as_tuple=False)
                causal_treatments = torch.masked_select(treatments, ts_causal_ids.bool())
                
                cate_effects = torch.repeat_interleave(effects, self.config.num_arms)

                if causal_idxs.nelement() == 0:
                    effects = torch.zeros(bs, self.config.num_arms)
                    return effects.cuda() if self.config.cuda else effects

                confounding_contexts = torch.stack([contexts[bidx, sidx] for bidx, sidx in causal_idxs])
                mu_pred, _ = self.estimator(confounding_contexts)

                causal_treatments = (~causal_treatments.bool()).long()
                causal_treatments = causal_treatments.unsqueeze(1)

                if len(mu_pred.shape) != len(causal_treatments.shape):
                    mu_pred = mu_pred.unsqueeze(1)

                mu_pred = mu_pred.gather(0, causal_treatments).squeeze()

                reverse_effects  = torch.zeros(bs, self.config.num_arms)
                if self.config.cuda:
                    reverse_effects = reverse_effects.cuda()

                reverse_effects[ts_causal_ids.bool()] = mu_pred
                reverse_effects = reverse_effects.flatten()
                
                cate_effects -= reverse_effects

                return cate_effects

        return deconfound
