from typing import Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque


import torch
import torch.nn as nn
from torch.nn.modules import utils

from torch.utils.data import DataLoader

from causal_env.envs import Timestep, TimestepDataset
from agents.base_agent import BaseAgent

import utils

import logging
logger = logging.getLogger(__name__)


@dataclass
class CausalAgentConfig:

    dim_in: Tuple[int] = (1, 28, 28)
    memsize: int = 100_000
    mc_samples: int = 100
    ensemble_size: int = 10
    dropout_rate: float = 0.5

    do_nothing: float = 1 # do nothing for this proportion of steps; exploration is off for the report findings
    cuda: bool = False
    batch_size:int = 32

    fixed_sigma: bool = False
    sigmoid_sigma: bool = False
    sigma: float = 1e-3

    lr: float  = 1e-3

    stochastic_acquisition: bool = False
    
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
        
        if self.config.stochastic_acquisition:
            unc = uncertaitnties.flatten()
            unc_probs = unc / unc.sum()
            unc_probs = utils.safenumpy(unc_probs)
            intervention = np.random.choice(unc_probs.shape[0], p=unc_probs)
            return intervention

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

        def deconfound(contexts, treatments,  ts_causal_ids, effects):
            with torch.no_grad():
                bs = len(contexts)

                causal_idxs = torch.nonzero(ts_causal_ids, as_tuple=False)
                causal_treatments = torch.masked_select(treatments, ts_causal_ids.bool())

                if causal_idxs.nelement() == 0:
                    effects = torch.zeros(bs, self.config.num_arms)
                    return effects.cuda() if self.config.cuda else effects

                confounding_contexts = torch.stack([contexts[bidx, sidx] for bidx, sidx in causal_idxs])
                mu_pred, _ = self.estimator(confounding_contexts)

                deconfounded_effects  = torch.zeros(bs, self.config.num_arms)

                if self.config.cuda:
                    deconfounded_effects = deconfounded_effects.cuda()

                if len(mu_pred.shape) != len(causal_treatments[None].shape):
                    mu_pred = mu_pred.unsqueeze(1)

                mu_pred = mu_pred.gather(0, causal_treatments[None]).squeeze()
  
                for i in range(bs):
                    total_batch_effect = torch.masked_select(mu_pred ,causal_idxs[:, 0] == i)
                    deconfounded_effects[i, ts_causal_ids[i]] = effects[i] - (total_batch_effect.sum() - total_batch_effect)

                return deconfounded_effects

        return deconfound